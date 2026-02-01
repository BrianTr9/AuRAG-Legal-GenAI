"""
Retrieval-Dependent Grammars (RDG) - Layer 2 (Research Edition v3)
==================================================================
Hybrid Implementation:
- Architecture: RDG2 (Concise, Modular)
- Logic: RDG.py (Mathematical Guarantee, Set Intersection Filter)
- prompt: RDG.py (Strict Reasoning -> Answer -> Citations)

Features:
1. Dynamic GBNF: Constraints built on-the-fly from retrieved context.
2. Structured CoT: Enforces 'Reasoning -> Answer -> Citations' flow.
3. Logical Consistency: Set(Retrieved) ∩ Set(Reasoning) ∩ Set(Generated).

Author: Trung Bao Truong
Date: February 2026
"""

import os
import re
import json
import sys
import io
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from langchain_core.documents import Document

# Public API exports
__all__ = [
    'Citation',
    'ReasoningStep',
    'StructuredReasoning',
    'StructuredOutput',
    'GBNFGrammarBuilder',
    'ReferenceExtractor',
    'RDGPipeline',
    'get_rdg_pipeline',
    'get_rdg_mode',
]

# ==========================================
# SETUP & CONFIG
# ==========================================
try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    LlamaGrammar = None

DEFAULT_MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
FALLBACK_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"

# ==========================================
# DATA MODELS
# ==========================================

@dataclass
class Citation:
    reference_id: str   # Full ID: doc_0_Section_I_TITLE
    section_num: str    # I, 2.1
    section_title: str  # TITLE
    contract_id: str    # doc_0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "section_num": self.section_num,
            "section_title": self.section_title
        }

    @staticmethod
    def from_reference_string(ref: str) -> 'Citation':
        section_marker = "_Section_"
        if section_marker not in ref:
            return Citation(ref, "Unknown", "Unknown", "Unknown")
        try:
            contract_and_prefix, rest = ref.split(section_marker, 1)
            if '_' in rest:
                section_num, section_title = rest.split('_', 1)
            else:
                section_num = rest
                section_title = "Unknown"
            return Citation(ref, section_num, section_title, contract_and_prefix)
        except:
            return Citation(ref, "Unknown", "Unknown", "Unknown")
    
    def to_short_id(self) -> str:
        """Compact ID for reasoning steps: doc_0_sec_I"""
        if self.contract_id.startswith("doc_"):
            parts = self.contract_id.split("_", 2)
            doc_prefix = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else self.contract_id
        else:
            doc_prefix = self.contract_id
        return f"{doc_prefix}_sec_{self.section_num}"
    
    @staticmethod
    def get_short_id_from_reference(ref: str) -> str:
        try:
            return Citation.from_reference_string(ref).to_short_id()
        except:
            return ref

@dataclass
class ReasoningStep:
    statement: str
    citations: List[str] = field(default_factory=list)
    step_type: str = "inference"
    
    def to_dict(self) -> Dict[str, Any]:
        return {"statement": self.statement, "citations": self.citations, "type": self.step_type}
    
    @staticmethod
    def from_dict(data: Dict) -> 'ReasoningStep':
        if not isinstance(data, dict): return ReasoningStep(str(data))
        return ReasoningStep(
            statement=data.get("statement", ""),
            citations=data.get("citations", []),
            step_type=data.get("type", "inference")
        )

@dataclass
class StructuredReasoning:
    steps: List[ReasoningStep]
    
    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}
    
    @staticmethod
    def from_dict(data: Dict) -> 'StructuredReasoning':
        if not isinstance(data, dict): return StructuredReasoning([])
        return StructuredReasoning([ReasoningStep.from_dict(s) for s in data.get("steps", [])])

@dataclass
class StructuredOutput:
    citations: List[Citation]
    reasoning: StructuredReasoning 
    answer: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "reasoning": self.reasoning.to_dict(),
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations]
        }
        if self.metadata: result["metadata"] = self.metadata
        return result

    @staticmethod
    def from_dict(data: Dict) -> 'StructuredOutput':
        citations = []
        for c in data.get("citations", []):
            try:
                if isinstance(c, dict):
                    if "reference_id" in c:
                        citations.append(Citation.from_reference_string(c["reference_id"]))
                    elif all(k in c for k in ["contract_id", "section_num", "section_title"]):
                        ref = f"{c['contract_id']}_Section_{c['section_num']}_{c['section_title']}"
                        citations.append(Citation.from_reference_string(ref))
                elif isinstance(c, str):
                    citations.append(Citation.from_reference_string(c))
            except: pass
        
        return StructuredOutput(
            citations=citations,
            reasoning=_parse_reasoning(data.get("reasoning", "")),
            answer=data.get("answer", "")
        )

# ==========================================
# GRAMMAR & HELPERS
# ==========================================

class GBNFGrammarBuilder:
    def build_structured_cot_grammar(self, valid_citations: List[str]) -> str:
        """Constructs GBNF grammar for Structured CoT."""
        if not valid_citations:
            full_enum = short_enum = '"\"" "no_citation" "\""'
        else:
            full_enum = ' | '.join([f'"\\"" "{c.replace("\\", "\\\\").replace('"', '\\"')}" "\\""' for c in valid_citations])
            # Short IDs for steps
            short_ids = [Citation.get_short_id_from_reference(c) for c in valid_citations]
            short_enum = ' | '.join([f'"\\"" "{c.replace("\\", "\\\\").replace('"', '\\"')}" "\\""' for c in short_ids])
        
        # Streamlined Grammar: Reasoning -> Answer (citations auto-collected from reasoning)
        return f'''
root ::= "{{" ws reasoning-field ws "," ws answer-field ws "}}"

reasoning-field ::= "\\"reasoning\\"" ws ":" ws reasoning-object
reasoning-object ::= "{{" ws "\\"steps\\"" ws ":" ws steps-array ws "}}"
steps-array ::= "[]" | "[" ws reasoning-step (ws "," ws reasoning-step)* ws "]"
reasoning-step ::= "{{" ws "\\"statement\\"" ws ":" ws string ws "," ws "\\"citations\\"" ws ":" ws short-citation-list ws "," ws "\\"type\\"" ws ":" ws step-type ws "}}"

short-citation-list ::= "[]" | "[" ws short-citation (ws "," ws short-citation)* ws "]"
short-citation ::= {short_enum}
step-type ::= "\\"premise\\"" | "\\"inference\\"" | "\\"conclusion\\""

answer-field ::= "\\"answer\\"" ws ":" ws string

string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrtu] [0-9a-fA-F]*
ws ::= [ \\t\\n]*
'''.strip()

class ReferenceExtractor:
    def extract_references(self, documents: List[Document]) -> List[str]:
        refs = []
        seen = set()
        for doc in documents:
            c_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
            s_num = doc.metadata.get('section_num', 'N/A')
            s_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
            ref = f"{c_id}_Section_{s_num}_{s_title}"
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
        return refs

def _parse_json_output(raw: str) -> Optional[Dict]:
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group()) if match else json.loads(raw)
    except: return None

def _parse_reasoning(data: Any) -> StructuredReasoning:
    if isinstance(data, dict): return StructuredReasoning.from_dict(data)
    if isinstance(data, str): return StructuredReasoning.from_dict({"steps": [{"statement": data}]})
    return StructuredReasoning([])

# ==========================================
# MAIN PIPELINE
# ==========================================

class RDGPipeline:
    def __init__(self, model_path: str = None, n_ctx: int = 16384, n_gpu_layers: int = -1, verbose: bool = False):
        if not LLAMA_CPP_AVAILABLE: raise RuntimeError("llama-cpp-python missing")
        
        self.model_path = model_path or (DEFAULT_MODEL_PATH if os.path.exists(DEFAULT_MODEL_PATH) else FALLBACK_MODEL_PATH)
        if not os.path.exists(self.model_path): raise RuntimeError("GGUF Model not found")
        
        print(f"\n🔧 Loading RDG Model: {os.path.basename(self.model_path)}")
        try:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            self.llm = Llama(model_path=self.model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=verbose)
        finally:
            sys.stderr = old_stderr
            
        self.grammar_builder = GBNFGrammarBuilder()
        self.extractor = ReferenceExtractor()
        print(f"✓ Model Loaded (Ctx: {n_ctx})")

    def _build_prompt(self, question: str, valid_refs: List[str], context_str: str) -> str:
        refs_display = "\n".join([f"  - {r}" for r in valid_refs])
        return f"""<|start_header_id|>system<|end_header_id|>

You are a precise legal document analyst. 

IMPORTANT RULES:
1. Use ONLY the citations listed in VALID_CITATIONS below.
2. Structure your response EXACTLY as the JSON format below.
3. Provide REASONING STEPS first, then the FINAL ANSWER.

<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
{context_str}

VALID_CITATIONS (Strictly constrained):
{refs_display}

QUESTION: {question}

Respond with this JSON structure:
{{
  "reasoning": {{
    "steps": [
      {{ "statement": "Analysis step 1...", "citations": ["doc_0_sec_I"], "type": "premise" }},
      {{ "statement": "Analysis step 2...", "citations": ["doc_0_sec_II"], "type": "inference" }}
    ]
  }},
  "answer": "Final concise answer here..."
}}

NOTE: 
- Use SHORT citation IDs (e.g., "doc_0_sec_I") in reasoning steps.
- Final citations list will be auto-generated from your reasoning steps.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def generate(self, question: str, documents: List[Document], max_tokens: int = 3072, temperature: float = 0) -> StructuredOutput:
        valid_refs = self.extractor.extract_references(documents)
        print(f"📋 Valid Citations: {len(valid_refs)}")
        
        if not valid_refs:
            return StructuredOutput([], StructuredReasoning([]), "No context available.")

        # 1. Build Grammar
        try:
            grammar_str = self.grammar_builder.build_structured_cot_grammar(valid_refs)
            grammar = LlamaGrammar.from_string(grammar_str)
        except Exception as e:
            return StructuredOutput([], StructuredReasoning([]), f"Grammar Error: {e}")

        # 2. Build Context & Prompt
        context_parts = []
        for doc in documents:
            c_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
            s_num = doc.metadata.get('section_num', 'N/A')
            s_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
            ref = f"{c_id}_Section_{s_num}_{s_title}"
            context_parts.append(f"[{ref}]\n{doc.page_content.strip()}")
        
        prompt = self._build_prompt(question, valid_refs, "\n\n---\n\n".join(context_parts))

        # 3. Generate
        print(f"🔄 RDG Thinking...", end="", flush=True)
        output = self.llm(
            prompt,
            grammar=grammar,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=1.12,  # Anti-loop
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        
        raw_text = output['choices'][0]['text'].strip()
        print(f"✓ Done ({len(raw_text)} chars)")

        # 4. Parse & Validate
        output_dict = _parse_json_output(raw_text)
        if not output_dict:
            return StructuredOutput([], StructuredReasoning([]), raw_text)

        # Basic validation
        reasoning = _parse_reasoning(output_dict.get('reasoning', ''))

        # --- LOGIC: Auto-collect final citations from reasoning ---
        # Goal: avoid omission when the model uses a short_id in reasoning but forgets to include
        # the corresponding full reference in the top-level "citations" list.
        #
        # We build a mapping short_id -> full reference(s) derived from valid_refs, then
        # reconstruct the final citations list in a stable, deterministic order.
        short_to_full_refs: Dict[str, List[str]] = {}
        for ref in valid_refs:
            short_id = Citation.get_short_id_from_reference(ref)
            short_to_full_refs.setdefault(short_id, []).append(ref)

        used_short_ids_in_order: List[str] = []
        seen_used_short = set()
        if isinstance(reasoning, StructuredReasoning) and reasoning.steps:
            for step in reasoning.steps:
                for short_id in (step.citations or []):
                    if short_id in seen_used_short:
                        continue
                    seen_used_short.add(short_id)
                    used_short_ids_in_order.append(short_id)

        full_refs_in_order: List[str] = []
        seen_full = set()
        for short_id in used_short_ids_in_order:
            for ref in short_to_full_refs.get(short_id, []):
                if ref in seen_full:
                    continue
                seen_full.add(ref)
                full_refs_in_order.append(ref)

        validated_citations = [Citation.from_reference_string(ref) for ref in full_refs_in_order]

        return StructuredOutput(
            citations=validated_citations,
            reasoning=reasoning,
            answer=output_dict.get('answer', '')
        )

# ==========================================
# SINGLETON EXPORT
# ==========================================
_rdg_instance = None
def get_rdg_pipeline(model_path: str = None, **kwargs):
    global _rdg_instance
    if _rdg_instance: return _rdg_instance
    _rdg_instance = RDGPipeline(model_path=model_path, **kwargs)
    return _rdg_instance

def get_rdg_mode() -> str: return 'gcd' if _rdg_instance else 'unknown'
