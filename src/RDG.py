"""
Retrieval-Dependent Grammars (RDG) - Layer 2
=============================================

Grammar-Constrained Decoding (GCD) via llama-cpp-python:
- JSON structure enforced at token level via GBNF grammar
- Citation validation via belt-and-suspenders post-check
- Mathematical guarantee of zero citation hallucination (syntactic)
- Requires GGUF model file

Reference Format (inherited from SPHR Layer 1):
    {contract_id}_Section_{section_num}_{section_title}

Author: Trung Bao (Brian) Truong
Date: January 2026
"""

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

import os
import re
import json
import sys
import io
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from langchain_core.documents import Document

# ==========================================
# LLAMA-CPP-PYTHON (Grammar-Constrained Decoding)
# ==========================================

try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    LlamaGrammar = None

# ==========================================
# CONFIGURATION
# ==========================================

# Default GGUF model paths (relative to project root)
DEFAULT_MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
FALLBACK_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"


# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class Citation:
    """Represents a citation with validation"""
    reference_id: str   # Full reference: doc_0_Section_I_TITLE
    section_num: str    # Section number: I, 2.1, 0
    section_title: str  # Section title: GENERAL_UNDERTAKING
    contract_id: str    # Contract ID: doc_0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (concise format)"""
        return {
            "contract_id": self.contract_id,
            "section_num": self.section_num,
            "section_title": self.section_title
        }

    @staticmethod
    def from_reference_string(ref: str) -> 'Citation':
        """Parse reference string into Citation object
        
        Expected format: {contract_id}_Section_{section_num}_{section_title}
        """
        section_marker = "_Section_"
        if section_marker not in ref:
            raise ValueError(f"Invalid reference format (missing '_Section_'): {ref}")
        
        contract_and_prefix, rest = ref.split(section_marker, 1)
        contract_id = contract_and_prefix
        
        if '_' not in rest:
            raise ValueError(f"Invalid reference format (malformed section part): {ref}")
        
        section_num, section_title = rest.split('_', 1)
        
        return Citation(
            reference_id=ref,
            section_num=section_num,
            section_title=section_title,
            contract_id=contract_id
        )
    
    def to_short_id(self) -> str:
        """Generate concise ID for reasoning steps."""
        if self.contract_id.startswith("doc_"):
            parts = self.contract_id.split("_", 2)
            if len(parts) >= 2:
                doc_prefix = f"{parts[0]}_{parts[1]}"
            else:
                doc_prefix = self.contract_id
        else:
            doc_prefix = self.contract_id
        
        return f"{doc_prefix}_sec_{self.section_num}"
    
    @staticmethod
    def get_short_id_from_reference(ref: str) -> str:
        """Extract short ID from full reference string"""
        try:
            citation = Citation.from_reference_string(ref)
            return citation.to_short_id()
        except:
            return ref


@dataclass
class ReasoningStep:
    """A single step in the chain of thought"""
    statement: str
    citations: List[str] = field(default_factory=list)
    step_type: str = "inference"  # "premise", "inference", "conclusion"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "citations": self.citations,
            "type": self.step_type
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'ReasoningStep':
        if not isinstance(data, dict):
            return ReasoningStep(statement=str(data) if data else "", step_type="inference")
        return ReasoningStep(
            statement=data.get("statement", ""),
            citations=data.get("citations", []),
            step_type=data.get("type", "inference")
        )


@dataclass
class StructuredReasoning:
    """Structured chain-of-thought reasoning"""
    steps: List[ReasoningStep]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with reasoning steps."""
        return {"steps": [step.to_dict() for step in self.steps]}
    
    @staticmethod
    def from_dict(data: Dict) -> 'StructuredReasoning':
        if data is None:
            return StructuredReasoning(steps=[])
        if isinstance(data, str):
            return StructuredReasoning(
                steps=[ReasoningStep(statement=data, step_type="inference")]
            )
        if not isinstance(data, dict):
            return StructuredReasoning(steps=[])
        steps_data = data.get("steps", [])
        if not isinstance(steps_data, list):
            steps_data = []
        steps = [ReasoningStep.from_dict(s) for s in steps_data]
        return StructuredReasoning(steps=steps)


@dataclass
class StructuredOutput:
    """Structured output with citations, reasoning, and answer"""
    citations: List[Citation]
    reasoning: StructuredReasoning 
    answer: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "reasoning": self.reasoning.to_dict(),
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations]
        }
        
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @staticmethod
    def from_dict(data: Dict) -> 'StructuredOutput':
        """Create StructuredOutput from dictionary."""
        citations = []
        
        for c in data.get("citations", []):
            try:
                if isinstance(c, dict):
                    if "reference_id" in c:
                        citations.append(Citation.from_reference_string(c["reference_id"]))
                    elif all(k in c for k in ["contract_id", "section_num", "section_title"]):
                        ref_id = f"{c['contract_id']}_Section_{c['section_num']}_{c['section_title']}"
                        citations.append(Citation.from_reference_string(ref_id))
                elif isinstance(c, str):
                    citations.append(Citation.from_reference_string(c))
            except (ValueError, KeyError):
                pass
        
        reasoning = _parse_reasoning(data.get("reasoning", ""))
        
        return StructuredOutput(
            citations=citations,
            reasoning=reasoning,
            answer=data.get("answer", "")
        )


# ==========================================
# GBNF GRAMMAR BUILDER (Core of True GCD)
# ==========================================

class GBNFGrammarBuilder:
    """
    Builds dynamic GBNF grammars for citation-constrained generation.
    """
    
    def build_grammar(self, valid_citations: List[str]) -> str:
        """
        Build grammar with REORDERED fields: Reasoning -> Answer -> Citations
        """
        if not valid_citations:
            citation_rule = 'citation-list ::= "[]"'
        else:
            escaped_refs = [
                f'"\\"" "{cit.replace("\\", "\\\\").replace('"', '\\"')}" "\\""'
                for cit in valid_citations
            ]
            citation_enum = ' | '.join(escaped_refs)
            citation_rule = f'''
citation-list ::= "[]" | "[" ws citation (ws "," ws citation)* ws "]"
citation ::= {citation_enum}
'''
        
        grammar = f'''
root ::= "{{" ws reasoning-field ws "," ws answer-field ws "," ws citations-field ws "}}"

reasoning-field ::= "\\"reasoning\\"" ws ":" ws string
answer-field ::= "\\"answer\\"" ws ":" ws string
citations-field ::= "\\"citations\\"" ws ":" ws {citation_rule.strip()}

string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrtu] [0-9a-fA-F]*
ws ::= [ \\t\\n]*
'''
        return grammar.strip()
    
    def build_structured_cot_grammar(self, valid_citations: List[str]) -> str:
        """
        Build Structured CoT grammar.
        Structure: Reasoning (Steps) -> Answer -> Citations (Summary)
        """
        if not valid_citations:
            full_citation_enum = '"\"" "no_citation" "\""'
            short_citation_enum = '"\"" "no_citation" "\""'
        else:
            # Full citation enum for top-level citations (Summary)
            escaped_full = []
            for cit in valid_citations:
                escaped = cit.replace('\\', '\\\\').replace('"', '\\"')
                escaped_full.append(f'"\\"" "{escaped}" "\\""')
            full_citation_enum = ' | '.join(escaped_full)
            
            # Short citation enum for step-level citations (Inference)
            escaped_short = []
            for cit in valid_citations:
                short_id = Citation.get_short_id_from_reference(cit)
                escaped = short_id.replace('\\', '\\\\').replace('"', '\\"')
                escaped_short.append(f'"\\"" "{escaped}" "\\""')
            short_citation_enum = ' | '.join(escaped_short)
        
        # Clean grammar: Reasoning -> Answer -> Citations (no duplicates)
        grammar = f'''
root ::= "{{" ws reasoning-field ws "," ws answer-field ws "," ws citations-field ws "}}"

reasoning-field ::= "\\"reasoning\\"" ws ":" ws reasoning-object
reasoning-object ::= "{{" ws "\\"steps\\"" ws ":" ws steps-array ws "}}"
steps-array ::= "[]" | "[" ws reasoning-step (ws "," ws reasoning-step)* ws "]"

reasoning-step ::= "{{" ws "\\"statement\\"" ws ":" ws string ws "," ws "\\"citations\\"" ws ":" ws short-citation-list ws "," ws "\\"type\\"" ws ":" ws step-type ws "}}"

short-citation-list ::= "[]" | "[" ws short-citation (ws "," ws short-citation)* ws "]"
short-citation ::= {short_citation_enum}

step-type ::= "\\"premise\\"" | "\\"inference\\"" | "\\"conclusion\\""

answer-field ::= "\\"answer\\"" ws ":" ws string

citations-field ::= "\\"citations\\"" ws ":" ws full-citation-list
full-citation-list ::= "[]" | "[" ws full-citation (ws "," ws full-citation)* ws "]"
full-citation ::= {full_citation_enum}

string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrtu] [0-9a-fA-F]*
ws ::= [ \\t\\n]*
'''
        return grammar.strip()


# ==========================================
# REFERENCE EXTRACTOR
# ==========================================

class ReferenceExtractor:
    def extract_references(self, documents: List[Document]) -> List[str]:
        references = []
        seen = set()
        for doc in documents:
            contract_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
            section_num = doc.metadata.get('section_num', 'N/A')
            section_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
            ref = f"{contract_id}_Section_{section_num}_{section_title}"
            if ref not in seen:
                references.append(ref)
                seen.add(ref)
        return references
    
    def extract_citations(self, documents: List[Document]) -> List[Citation]:
        refs = self.extract_references(documents)
        return [Citation.from_reference_string(ref) for ref in refs]


# ==========================================
# JSON PARSING HELPER
# ==========================================

def _parse_json_output(raw_text: str) -> Optional[Dict[str, Any]]:
    try:
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None

def _parse_reasoning(reasoning_data: Any) -> StructuredReasoning:
    if isinstance(reasoning_data, dict):
        return StructuredReasoning.from_dict(reasoning_data)
    elif isinstance(reasoning_data, str):
        return StructuredReasoning.from_dict(reasoning_data)
    return StructuredReasoning(steps=[])


# ==========================================
# RDG PIPELINE - TRUE GCD IMPLEMENTATION
# ==========================================

class RDGPipeline:
    def __init__(self, model_path: str = None, n_ctx: int = 8192, n_gpu_layers: int = -1, verbose: bool = False):
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python not installed.")
        
        self.model_path = self._find_model(model_path)
        print(f"\n🔧 Loading GGUF model: {os.path.basename(self.model_path)}")
        
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            self.llm = Llama(
                model_path=self.model_path, 
                n_ctx=n_ctx, 
                n_gpu_layers=n_gpu_layers, 
                verbose=False
            )
        finally:
            sys.stderr = old_stderr
        
        print(f"✓ Model ready (context: {n_ctx} tokens)")
        self.grammar_builder = GBNFGrammarBuilder()
        self.reference_extractor = ReferenceExtractor()
    
    def _find_model(self, model_path: Optional[str]) -> str:
        if model_path and os.path.exists(model_path): return model_path
        candidates = [DEFAULT_MODEL_PATH, FALLBACK_MODEL_PATH]
        for path in candidates:
            if os.path.exists(path): return path
        raise FileNotFoundError("No GGUF model found.")
    
    def _build_context(self, documents: List[Document]) -> str:
        context_parts = []
        for doc in documents:
            content = (doc.page_content or '').strip()
            contract_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
            sec_num = doc.metadata.get('section_num', 'N/A')
            sec_title = (doc.metadata.get('section_title') or 'Unknown').strip()
            ref = f"{contract_id}_Section_{sec_num}_{sec_title.replace(' ', '_')}"
            context_parts.append(f"[{ref}]\n{content}")
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, valid_refs: List[str]) -> str:
        refs_display = "\n".join([f"  - {ref}" for ref in valid_refs])
        
        # UPDATED PROMPT: Enforces Reasoning -> Answer -> Citations
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a precise legal document analyst. 

IMPORTANT RULES:
1. Use ONLY the citations listed in VALID_CITATIONS below.
2. Structure your response EXACTLY as the JSON format below.
3. First provide the REASONING STEPS, then the FINAL ANSWER, and finally the LIST OF CITATIONS used.
4. **NO DUPLICATE CITATIONS** in the final list.

<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
{context}

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
  "answer": "Final concise answer here...",
  "citations": [
    "doc_0_Section_I_TITLE",
    "doc_0_Section_II_TITLE"
  ]
}}

NOTE: 
- The "citations" list must summarize the IDs used in reasoning.
- DO NOT duplicate IDs in the "citations" list.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
    
    def generate(self, question: str, documents: List[Document], max_tokens: int = 1024, temperature: float = 0.1, use_grammar: bool = True) -> StructuredOutput:
        valid_refs = self.reference_extractor.extract_references(documents)
        print(f"📋 Valid citations from retrieval: {len(valid_refs)}")
        
        if not valid_refs:
            return StructuredOutput(citations=[], reasoning=StructuredReasoning(steps=[]), answer="No context available.")
        
        grammar = None
        if use_grammar:
            try:
                # Build structured CoT grammar: Reasoning -> Answer -> Citations
                grammar_str = self.grammar_builder.build_structured_cot_grammar(valid_refs)
                grammar = LlamaGrammar.from_string(grammar_str)
                print("✓ GBNF grammar compiled (Reasoning -> Answer -> Citations)")
            except Exception as e:
                print(f"   ⚠️  Grammar error: {e}")
                grammar = None
        
        context = self._build_context(documents)
        prompt = self._build_prompt(question, context, valid_refs)
        
        print(f"🔄 Generating answer...", end="", flush=True)
        
        # === KEY FIX: Added repeat_penalty ===
        output = self.llm(
            prompt,
            grammar=grammar,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=1.12,  # <--- CRITICAL: Prevents list loops
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        
        raw_text = output['choices'][0]['text'].strip()
        print(f"✓ Generated {len(raw_text)} characters")
        
        output_dict = _parse_json_output(raw_text)
        if output_dict is None:
            return StructuredOutput(citations=[], reasoning=StructuredReasoning(steps=[]), answer=raw_text)
        
        reasoning = _parse_reasoning(output_dict.get('reasoning', ''))
        validated_citations = self._validate_citations(output_dict.get('citations', []), valid_refs)
        
        # === MATHEMATICAL GUARANTEE: Set Intersection Filter ===
        # Only include citations that satisfy: Set(Retrieved) ∩ Set(Reasoning) ∩ Set(Generated)
        # 1. Set(Retrieved) = valid_refs (already enforced by _validate_citations)
        # 2. Set(Reasoning) = citations used in reasoning.steps[].citations
        # 3. Set(Generated) = validated_citations
        
        if isinstance(reasoning, StructuredReasoning) and reasoning.steps:
            # Build set of citation IDs used in reasoning (both short & full formats)
            used_citation_ids = set()
            for step in reasoning.steps:
                used_citation_ids.update(step.citations)
            
            # Triệt để filter: ONLY keep citations that appear in reasoning
            # No fallback - if not used in reasoning, not in final list
            filtered_citations = []
            for cit in validated_citations:
                if cit.to_short_id() in used_citation_ids or cit.reference_id in used_citation_ids:
                    filtered_citations.append(cit)
            validated_citations = filtered_citations
        else:
            # Edge case: No reasoning steps = No citations (strict enforcement)
            validated_citations = []
        
        return StructuredOutput(
            citations=validated_citations,
            reasoning=reasoning,
            answer=output_dict.get('answer', '')
        )
    
    def _validate_citations(self, raw_citations: List[Any], valid_refs: List[str]) -> List[Citation]:
        """Validate and deduplicate citations against retrieved references.
        
        This is the single source of truth for citation deduplication.
        """
        validated = []
        seen = set()
        for cit in raw_citations:
            cit_str = str(cit)
            if cit_str in valid_refs and cit_str not in seen:
                try:
                    validated.append(Citation.from_reference_string(cit_str))
                    seen.add(cit_str)
                except ValueError:
                    continue  # Skip invalid citations
        return validated
    
    def __call__(self, question: str, documents: List[Document], max_tokens: int = 1024) -> StructuredOutput:
        return self.generate(question, documents, max_tokens)

# ==========================================
# SINGLETON
# ==========================================
_rdg_instance = None
def get_rdg_pipeline(model_path: str = None, **kwargs):
    global _rdg_instance
    if _rdg_instance: return _rdg_instance
    
    gguf_path = model_path
    if not gguf_path:
        if os.path.exists(DEFAULT_MODEL_PATH): gguf_path = DEFAULT_MODEL_PATH
        elif os.path.exists(FALLBACK_MODEL_PATH): gguf_path = FALLBACK_MODEL_PATH
    
    if not gguf_path or not os.path.exists(gguf_path):
        raise RuntimeError(f"Model not found at {DEFAULT_MODEL_PATH}")
        
    _rdg_instance = RDGPipeline(model_path=gguf_path, **kwargs)
    return _rdg_instance

def get_rdg_mode() -> str:
    return 'gcd' if _rdg_instance else 'unknown'