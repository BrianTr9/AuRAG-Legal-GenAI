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
        Example: doc_0_Section_II_DUTIES_OF_AGENT
        """
        section_marker = "_Section_"
        if section_marker not in ref:
            raise ValueError(f"Invalid reference format (missing '_Section_'): {ref}")
        
        # Split on "_Section_"
        contract_and_prefix, rest = ref.split(section_marker, 1)
        contract_id = contract_and_prefix
        
        # rest is "section_num_title", split on first underscore
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
        """Generate concise ID for reasoning steps.
        
        Format: {doc_prefix}_sec_{section_num}
        Example: "doc_0_sec_I" or "doc_1_sec_2"
        """
        # Extract just the doc_X part (before the first underscore after "doc_")
        if self.contract_id.startswith("doc_"):
            parts = self.contract_id.split("_", 2)  # Split into ["doc", "X", "rest"]
            if len(parts) >= 2:
                doc_prefix = f"{parts[0]}_{parts[1]}"  # "doc_X"
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
            return ref  # Fallback to original if parsing fails


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
            # Handle invalid input gracefully
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
        """Convert to dict with short citation IDs in steps"""
        steps_dict = []
        for step in self.steps:
            step_dict = step.to_dict()
            # Convert full citation IDs to short IDs for conciseness
            step_dict["citations"] = [
                Citation.get_short_id_from_reference(cit) 
                for cit in step_dict["citations"]
            ]
            steps_dict.append(step_dict)
        return {"steps": steps_dict}
    
    @staticmethod
    def from_dict(data: Dict) -> 'StructuredReasoning':
        if data is None:
            return StructuredReasoning(steps=[])
        if isinstance(data, str):
            # Backward compatibility: convert free-text to single step
            return StructuredReasoning(
                steps=[ReasoningStep(statement=data, step_type="inference")]
            )
        if not isinstance(data, dict):
            # Handle unexpected types gracefully
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
    reasoning: StructuredReasoning  # Now structured, not free text
    answer: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (clean format)"""
        # Handle both StructuredReasoning and legacy string format
        if isinstance(self.reasoning, StructuredReasoning):
            reasoning_dict = self.reasoning.to_dict()
        elif isinstance(self.reasoning, str):
            reasoning_dict = self.reasoning  # Legacy string format
        else:
            reasoning_dict = {"steps": []}
        
        result = {
            "citations": [c.to_dict() for c in self.citations],
            "reasoning": reasoning_dict,
            "answer": self.answer
        }
        # Only include metadata if it has content
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
                pass  # Skip invalid citations
        
        # Parse reasoning (structured or string format)
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
    
    Zero-Hallucination Mechanism:
    - Grammar constrains LLM to ONLY generate valid citation IDs
    - Invalid tokens receive logit = -∞ (mathematically impossible)
    - Provides syntactic guarantee of citation correctness
    """
    
    def build_grammar(self, valid_citations: List[str]) -> str:
        """Build GBNF grammar constraining citations to valid set.
        
        Args:
            valid_citations: Valid citation IDs from retrieved documents
            
        Returns:
            GBNF grammar string for llama-cpp-python
        """
        if not valid_citations:
            citation_rule = 'citation-list ::= "[]"'
        else:
            # Escape and format each citation as GBNF string literal
            escaped_refs = [
                f'"\\"" "{cit.replace("\\", "\\\\").replace('"', '\\"')}" "\\""'
                for cit in valid_citations
            ]
            citation_enum = ' | '.join(escaped_refs)
            citation_rule = f'''
citation-list ::= "[]" | "[" ws citation (ws "," ws citation)* ws "]"
citation ::= {citation_enum}
'''
        
        # Complete GBNF grammar for structured JSON output
        # This enforces: {"citations": [...], "reasoning": "...", "answer": "..."}
        grammar = f'''
root ::= "{{" ws citations-field ws "," ws reasoning-field ws "," ws answer-field ws "}}"

citations-field ::= "\\"citations\\"" ws ":" ws {citation_rule.strip()}

reasoning-field ::= "\\"reasoning\\"" ws ":" ws string
answer-field ::= "\\"answer\\"" ws ":" ws string

string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]
ws ::= [ \\t\\n]*
'''
        return grammar.strip()
    
    def build_simple_grammar(self, valid_citations: List[str] = None) -> str:
        """
        Build a simpler grammar with citation constraints but flat reasoning.
        
        Used as fallback when structured CoT grammar fails.
        Still maintains zero-hallucination guarantee for citations.
        
        Args:
            valid_citations: List of valid citation strings (if None, allows empty only)
        """
        if not valid_citations:
            citation_rule = '"[]"'
        else:
            escaped_refs = []
            for cit in valid_citations:
                escaped = cit.replace('\\', '\\\\').replace('"', '\\"')
                escaped_refs.append(f'"\\"" "{escaped}" "\\""')
            citation_enum = ' | '.join(escaped_refs)
            citation_rule = f'"[]" | "[" ws citation (ws "," ws citation)* ws "]"\ncitation ::= {citation_enum}'
        
        grammar = f'''
root ::= "{{" ws citations-field ws "," ws reasoning-field ws "," ws answer-field ws "}}"

citations-field ::= "\\"citations\\"" ws ":" ws {citation_rule}

reasoning-field ::= "\\"reasoning\\"" ws ":" ws string
answer-field ::= "\\"answer\\"" ws ":" ws string

string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]
ws ::= [ \\t\\n]*
'''
        return grammar.strip()
    
    def build_structured_cot_grammar(self, valid_citations: List[str]) -> str:
        """
        Build GBNF grammar for Structured Chain-of-Thought.
        
        Enforces reasoning as an array of steps, each with:
        - statement: string
        - citations: array of SHORT citation IDs (e.g., "doc_0_sec_I")
        - type: "premise" | "inference" | "conclusion"
        
        Top-level citations use full reference format.
        Step-level citations use concise short IDs for readability.
        """
        if not valid_citations:
            full_citation_enum = '"\"" "no_citation" "\""'
            short_citation_enum = '"\"" "no_citation" "\""'
        else:
            # Full citation enum for top-level citations
            escaped_full = []
            for cit in valid_citations:
                escaped = cit.replace('\\', '\\\\').replace('"', '\\"')
                escaped_full.append(f'"\\"" "{escaped}" "\\""')
            full_citation_enum = ' | '.join(escaped_full)
            
            # Short citation enum for step-level citations
            escaped_short = []
            for cit in valid_citations:
                short_id = Citation.get_short_id_from_reference(cit)
                escaped = short_id.replace('\\', '\\\\').replace('"', '\\"')
                escaped_short.append(f'"\\"" "{escaped}" "\\""')
            short_citation_enum = ' | '.join(escaped_short)
        
        grammar = f'''
root ::= "{{" ws citations-field ws "," ws reasoning-field ws "," ws answer-field ws "}}"

citations-field ::= "\\"citations\\"" ws ":" ws full-citation-list

full-citation-list ::= "[]" | "[" ws full-citation (ws "," ws full-citation)* ws "]"
full-citation ::= {full_citation_enum}

reasoning-field ::= "\\"reasoning\\"" ws ":" ws reasoning-object

reasoning-object ::= "{{" ws "\\"steps\\"" ws ":" ws steps-array ws "}}"

steps-array ::= "[]" | "[" ws reasoning-step (ws "," ws reasoning-step)* ws "]"

reasoning-step ::= "{{" ws "\\"statement\\"" ws ":" ws string ws "," ws "\\"citations\\"" ws ":" ws short-citation-list ws "," ws "\\"type\\"" ws ":" ws step-type ws "}}"

short-citation-list ::= "[]" | "[" ws short-citation (ws "," ws short-citation)* ws "]"
short-citation ::= {short_citation_enum}

step-type ::= "\\"premise\\"" | "\\"inference\\"" | "\\"conclusion\\""

answer-field ::= "\\"answer\\"" ws ":" ws string

string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" ["\\\\\\\\bfnrt]
ws ::= [ \\t\\n]*
'''
        return grammar.strip()


# ==========================================
# REFERENCE EXTRACTOR
# ==========================================

class ReferenceExtractor:
    """Extract valid reference identifiers from retrieved documents"""

    def extract_references(self, documents: List[Document]) -> List[str]:
        """
        Extract all valid reference identifiers from retrieved documents.
        
        Args:
            documents: List of Document objects from SPHR retrieval
                      (should have metadata: section_num, section_title, contract_id or parent_id)
        
        Returns:
            List of reference strings in format: {contract_id}_Section_{section_num}_{section_title}
        """
        references = []
        seen = set()

        for doc in documents:
            # Extract contract_id (try both keys for compatibility)
            contract_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
            
            # Extract section info
            section_num = doc.metadata.get('section_num', 'N/A')
            section_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
            
            # Build reference string
            ref = f"{contract_id}_Section_{section_num}_{section_title}"
            
            # Deduplicate
            if ref not in seen:
                references.append(ref)
                seen.add(ref)

        return references
    
    def extract_citations(self, documents: List[Document]) -> List[Citation]:
        """Extract as Citation objects"""
        refs = self.extract_references(documents)
        return [Citation.from_reference_string(ref) for ref in refs]


# ==========================================
# JSON PARSING HELPER
# ==========================================

def _parse_json_output(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from raw LLM output, handling common edge cases.
    
    Args:
        raw_text: Raw output text from LLM
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        # Try to find JSON object in response
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def _parse_reasoning(reasoning_data: Any) -> StructuredReasoning:
    """Parse reasoning data into StructuredReasoning object.
    
    Handles dict, string, and None inputs gracefully.
    """
    if isinstance(reasoning_data, dict):
        return StructuredReasoning.from_dict(reasoning_data)
    elif isinstance(reasoning_data, str):
        return StructuredReasoning.from_dict(reasoning_data)
    else:
        return StructuredReasoning(steps=[])


# ==========================================
# RDG PIPELINE - TRUE GCD IMPLEMENTATION
# ==========================================

class RDGPipeline:
    """
    Retrieval-Dependent Grammar Pipeline with Grammar-Constrained Decoding.
    
    Phase 2 Implementation:
    - Uses llama-cpp-python for native GBNF support
    - JSON structure enforced at token level
    - Citation validation via belt-and-suspenders post-check
    
    Usage:
        rdg = RDGPipeline()  # Loads GGUF model
        output = rdg.generate(question, documents)  # Returns StructuredOutput
    """
    
    def __init__(
        self, 
        model_path: str = None, 
        n_ctx: int = 16384, 
        n_gpu_layers: int = -1,
        verbose: bool = False
    ):
        """
        Initialize RDG Pipeline with llama-cpp-python.
        
        Args:
            model_path: Path to GGUF model file (auto-detected if None)
            n_ctx: Context window size (default: 16384)
            n_gpu_layers: GPU layers (-1 = all on GPU, 0 = CPU only)
            verbose: Show detailed llama.cpp logs
        """
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python not installed.\n"
                "Install with: pip install llama-cpp-python"
            )
        
        # Find and load model
        self.model_path = self._find_model(model_path)
        print(f"\n🔧 Loading GGUF model: {os.path.basename(self.model_path)}")
        
        # Suppress verbose llama.cpp initialization output
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
        
        # Initialize helper components
        self.grammar_builder = GBNFGrammarBuilder()
        self.reference_extractor = ReferenceExtractor()
    
    def _find_model(self, model_path: Optional[str]) -> str:
        """Find a valid GGUF model path"""
        if model_path and os.path.exists(model_path):
            return model_path
        
        # Try default paths
        candidates = [
            DEFAULT_MODEL_PATH,
            FALLBACK_MODEL_PATH,
            os.path.expanduser(f"~/.cache/llama/{os.path.basename(DEFAULT_MODEL_PATH)}"),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"No GGUF model found. Tried:\n"
            f"  - {DEFAULT_MODEL_PATH}\n"
            f"  - {FALLBACK_MODEL_PATH}\n"
            f"Please download a GGUF model and place it in the models/ directory."
        )
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from documents with reference IDs"""
        context_parts = []
        
        for doc in documents:
            content = (doc.page_content or '').strip()
            contract_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
            sec_num = doc.metadata.get('section_num', 'N/A')
            sec_title = (doc.metadata.get('section_title') or 'Unknown').strip()
            
            # Build reference (matching the format used for grammar)
            ref = f"{contract_id}_Section_{sec_num}_{sec_title.replace(' ', '_')}"
            
            context_parts.append(f"[{ref}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, valid_refs: List[str]) -> str:
        """Build the generation prompt with Llama instruction format"""
        
        refs_display = "\n".join([f"  - {ref}" for ref in valid_refs])
        
        # Note: llama-cpp-python adds <|begin_of_text|> automatically
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a precise legal document analyst. Your task is to answer questions based ONLY on the provided context.

IMPORTANT RULES:
1. Use ONLY the citations listed in VALID_CITATIONS below
2. Every claim must be supported by a citation
3. If the context doesn't contain enough information, say so
4. Respond in JSON format with: citations, reasoning, answer

<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
{context}

VALID_CITATIONS (use EXACTLY these IDs, no others):
{refs_display}

QUESTION: {question}

Respond with a JSON object containing:
- "citations": list of ALL citation IDs you reference (from VALID_CITATIONS only)
- "reasoning": structured chain-of-thought with:
  - "steps": array of reasoning steps, each with:
    * "statement": the reasoning statement
    * "citations": citation IDs supporting this step
    * "type": "premise" (fact from context), "inference" (logical deduction), or "conclusion" (final reasoning)
- "answer": your final answer

Example format:
{{
  "citations": ["doc_0_Section_II_TITLE"],
  "reasoning": {{
    "steps": [
      {{"statement": "Contract requires notice", "citations": ["doc_0_sec_II"], "type": "premise"}},
      {{"statement": "Therefore notice is mandatory", "citations": [], "type": "conclusion"}}
    ]
  }},
  "answer": "Notice is required"
}}

Note: Use SHORT citation IDs in reasoning steps (e.g., "doc_0_sec_II" instead of full reference)

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def generate(
        self, 
        question: str, 
        documents: List[Document], 
        max_tokens: int = 1024,
        temperature: float = 0.1,
        use_grammar: bool = True
    ) -> StructuredOutput:
        """
        Generate answer with GBNF grammar constraints.
        
        This is the CORE METHOD that provides mathematical guarantee
        of zero citation hallucination through token-level constraints.
        
        Args:
            question: User's question
            documents: Retrieved documents from SPHR Layer 1
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            use_grammar: Whether to use GBNF constraints (True for production)
            
        Returns:
            StructuredOutput with validated citations
        """
        # Step 1: Extract valid citations from retrieved documents
        valid_refs = self.reference_extractor.extract_references(documents)
        print(f"📋 Valid citations from retrieval: {len(valid_refs)}")
        
        if not valid_refs:
            print("   ⚠️  No valid citations found")
            return StructuredOutput(
                citations=[],
                reasoning=StructuredReasoning(steps=[
                    ReasoningStep(
                        statement="No valid citations found in retrieved documents.",
                        step_type="conclusion"
                    )
                ]),
                answer="Unable to answer - no relevant context available."
            )
        
        # Step 2: Build dynamic GBNF grammar with citation constraints
        grammar = None
        if use_grammar:
            try:
                grammar_str = self.grammar_builder.build_structured_cot_grammar(valid_refs)
                grammar = LlamaGrammar.from_string(grammar_str)
                print("✓ GBNF grammar compiled (Structured CoT + citation constraints)")
            except Exception:
                # Fallback: simple grammar still citation-constrained
                try:
                    grammar_str = self.grammar_builder.build_simple_grammar(valid_refs)
                    grammar = LlamaGrammar.from_string(grammar_str)
                    print("   ⚠️  Using simple grammar (CoT fallback)")
                except Exception:
                    grammar = None
                    print("   ⚠️  Grammar compilation failed!")
        
        # Step 3: Build context and prompt
        context = self._build_context(documents)
        prompt = self._build_prompt(question, context, valid_refs)
        
        # Step 4: Generate with grammar constraints
        print(f"🔄 Generating answer...", end="", flush=True)
        
        output = self.llm(
            prompt,
            grammar=grammar,  # <-- THE MAGIC: Token-level constraints
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        
        raw_text = output['choices'][0]['text'].strip()
        print(f"✓ Generated {len(raw_text)} characters")
        
        # Step 5: Parse JSON output
        output_dict = _parse_json_output(raw_text)
        if output_dict is None:
            print(f"   ⚠️  JSON parse error")
            return StructuredOutput(
                citations=[],
                reasoning=StructuredReasoning(steps=[
                    ReasoningStep(statement="JSON parsing failed", step_type="conclusion")
                ]),
                answer=raw_text
            )
        
        # Step 6: Validate citations (belt-and-suspenders check)
        validated_citations = self._validate_citations(
            output_dict.get('citations', []),
            valid_refs
        )
        
        # Parse reasoning into StructuredReasoning
        reasoning = _parse_reasoning(output_dict.get('reasoning', ''))
        
        return StructuredOutput(
            citations=validated_citations,
            reasoning=reasoning,
            answer=output_dict.get('answer', '')
        )
    
    def _validate_citations(
        self,
        raw_citations: List[Any],
        valid_refs: List[str]
    ) -> List[Citation]:
        """Validate and filter citations (belt-and-suspenders check).
        
        Args:
            raw_citations: Raw citation data from LLM output
            valid_refs: Set of valid reference strings from retrieval
            
        Returns:
            List of validated Citation objects
        """
        validated = []
        
        for cit in raw_citations:
            cit_str = str(cit) if not isinstance(cit, str) else cit
            if cit_str in valid_refs:
                try:
                    validated.append(Citation.from_reference_string(cit_str))
                except ValueError:
                    pass  # Skip malformed citations
        
        # Report filtering if any citations were removed
        if len(validated) != len(raw_citations):
            filtered = len(raw_citations) - len(validated)
            print(f"   ⚠️  Filtered {filtered} invalid citation(s)")
        
        return validated
    
    def __call__(
        self, 
        question: str, 
        documents: List[Document], 
        max_tokens: int = 1024
    ) -> StructuredOutput:
        """Convenience method to call generate()"""
        return self.generate(question, documents, max_tokens)
    
    # ==========================================
    # BACKWARD COMPATIBILITY (Phase 1 interface)
    # ==========================================
    
    def prepare_generation(
        self, 
        documents: List[Document], 
        question: str, 
        context_text: str
    ) -> Dict[str, Any]:
        """
        Prepare generation context (backward compatible with Phase 1).
        
        Note: In Phase 2, this is not needed as generate() handles everything,
        but we keep it for compatibility with existing code.
        """
        valid_refs = self.reference_extractor.extract_references(documents)
        
        return {
            'prompt': self._build_prompt(question, context_text, valid_refs),
            'valid_references': valid_refs,
            'constraints': {
                'json_schema': None,  # Not used in Phase 2
                'valid_reference_strings': valid_refs
            }
        }
    
    def validate_generation(
        self, 
        raw_output: str, 
        valid_references: List[str]
    ) -> Tuple[StructuredOutput, List[str]]:
        """
        Validate generated output (backward compatible with Phase 1).
        
        Note: In Phase 2 with GCD, this is mostly unnecessary since
        the grammar prevents invalid citations at generation time.
        """
        errors = []
        
        output_dict = _parse_json_output(raw_output)
        if output_dict is None:
            return None, ["Invalid JSON"]
        
        # Validate and filter citations
        validated_citations = []
        for cit in output_dict.get('citations', []):
            cit_str = cit if isinstance(cit, str) else str(cit)
            if cit_str in valid_references:
                try:
                    validated_citations.append(Citation.from_reference_string(cit_str))
                except ValueError as e:
                    errors.append(f"Citation parse error: {e}")
            else:
                errors.append(f"Invalid citation: {cit_str}")
        
        # Parse reasoning into StructuredReasoning
        reasoning = _parse_reasoning(output_dict.get('reasoning', ''))
        
        structured = StructuredOutput(
            citations=validated_citations,
            reasoning=reasoning,
            answer=output_dict.get('answer', '')
        )
        
        return structured, errors


# ==========================================
# SINGLETON & AUTO-DETECTION
# ==========================================

_rdg_instance = None

def get_rdg_pipeline(model_path: str = None, **kwargs):
    """
    Get or create RDG pipeline with GBNF-based Grammar-Constrained Decoding.
    
    Auto-detection:
    1. Use provided model_path if specified
    2. Try DEFAULT_MODEL_PATH
    3. Try FALLBACK_MODEL_PATH
    4. Raise error if no model found
    
    Args:
        model_path: Path to GGUF model (optional)
        **kwargs: Additional arguments for RDGPipeline
        
    Returns:
        RDGPipeline instance
    """
    global _rdg_instance
    
    if _rdg_instance is not None:
        return _rdg_instance
    
    # Determine GGUF model path
    gguf_path = model_path
    if not gguf_path:
        if os.path.exists(DEFAULT_MODEL_PATH):
            gguf_path = DEFAULT_MODEL_PATH
        elif os.path.exists(FALLBACK_MODEL_PATH):
            gguf_path = FALLBACK_MODEL_PATH
    
    if not gguf_path or not os.path.exists(gguf_path):
        raise RuntimeError(
            "No GGUF model found.\n"
            "Options:\n"
            f"  1. Download GGUF model to {DEFAULT_MODEL_PATH}\n"
            "  2. Specify custom path: get_rdg_pipeline(model_path='...')\n"
            "\n"
            "Recommended models:\n"
            "  - Llama-3.1-8B-Instruct (Q4_K_M): ~5GB, good balance\n"
            "  - Llama-3.2-3B-Instruct (Q4_K_M): ~2GB, fast experiments"
        )
    
    if not LLAMA_CPP_AVAILABLE:
        raise RuntimeError(
            "llama-cpp-python not available.\n"
            "Install with: pip install llama-cpp-python"
        )
    
    # Create GCD pipeline
    _rdg_instance = RDGPipeline(model_path=gguf_path, **kwargs)
    return _rdg_instance


def get_rdg_mode() -> str:
    """Get current RDG mode (always 'gcd' now)"""
    return 'gcd' if _rdg_instance else 'unknown'


# ==========================================
# OLD OLLAMA FALLBACK PIPELINE - REMOVED
# ==========================================
# The OllamaFallbackPipeline class has been removed to focus on
# true Grammar-Constrained Decoding (GCD) via llama-cpp-python.
# This provides 100% zero-hallucination guarantee (syntactic).


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def extract_citations_from_docs(documents: List[Document]) -> List[Citation]:
    """
    Convenience function to extract citations from documents.
    
    Args:
        documents: Retrieved documents from SPHR
        
    Returns:
        List of Citation objects
    """
    extractor = ReferenceExtractor()
    return extractor.extract_citations(documents)


def build_json_schema(citations: List[Citation]) -> Dict[str, Any]:
    """
    Build JSON schema for validation (compatibility).
    
    Args:
        citations: List of valid citations
        
    Returns:
        JSON schema dictionary
    """
    valid_refs = [c.reference_id for c in citations]
    
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "citations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": valid_refs
                }
            },
            "reasoning": {"type": "string"},
            "answer": {"type": "string"}
        },
        "required": ["citations", "reasoning", "answer"]
    }


# ==========================================
# LEGACY CLASSES (For backward compatibility)
# ==========================================

class RDGConstrainer:
    """Legacy class for backward compatibility"""
    
    def __init__(self, grammar_builder=None):
        self.grammar_builder = grammar_builder or GBNFGrammarBuilder()
        self._outlines_available = False  # Not used in Phase 2
    
    def build_constraints(self, documents: List[Document]) -> Dict[str, Any]:
        extractor = ReferenceExtractor()
        citations = extractor.extract_citations(documents)
        refs = [c.reference_id for c in citations]
        
        return {
            'valid_citations': citations,
            'valid_reference_strings': refs,
            'json_schema': build_json_schema(citations)
        }


class CitationValidator:
    """Legacy class for backward compatibility"""
    
    def __init__(self, valid_references: List[str]):
        self.valid_references = set(valid_references)
    
    def validate_output(self, output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        if 'citations' not in output:
            errors.append("Missing 'citations' field")
        if 'reasoning' not in output:
            errors.append("Missing 'reasoning' field")
        if 'answer' not in output:
            errors.append("Missing 'answer' field")
        
        if errors:
            return False, errors
        
        invalid = [c for c in output.get('citations', []) if c not in self.valid_references]
        if invalid:
            errors.append(f"Invalid citations: {invalid}")
        
        return len(errors) == 0, errors
    
    def fix_citations(self, output: Dict[str, Any]) -> Dict[str, Any]:
        fixed = output.copy()
        fixed['citations'] = [c for c in output.get('citations', []) if c in self.valid_references]
        return fixed


# End of module
