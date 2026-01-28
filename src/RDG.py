"""
Retrieval-Dependent Grammars (RDG) - Layer 2
=============================================

This module implements Retrieval-Dependent Grammars for citation-aware, hallucination-free generation.

Key Concepts:
- RDG extends Input-Dependent Grammars (IDG) by conditioning not just on input query,
  but on the actual retrieved documents.
- Grammar is dynamically constructed based on retrieved references.
- Constrains LLM to generate citations only from the retrieved set.
- Output is structured JSON: {Citation, Reasoning, Answer}

Architecture:
1. Extract reference identifiers from retrieved documents
2. Build dynamic GBNF grammar to restrict citations to valid references
3. Generate constrained JSON output via Outlines or token masking
4. Enforce structured Chain-of-Thought for transparency

Reference Format (inherited from SPHR Layer 1):
    {contract_id}_Section_{section_num}_{section_title}
    Examples:
    - "doc_0_Section_I_GENERAL_UNDERTAKING"
    - "doc_1_Section_2.1_Compensation"
    - "doc_0_Section_0_Preamble"

Author: Trung Bao (Brian) Truong
Date: January 2026
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from langchain_core.documents import Document


# ==========================================
# DATA STRUCTURES
# ==========================================

class CitationFormat(Enum):
    """Citation format types"""
    FULL = "full"  # doc_0_Section_I_GENERAL_UNDERTAKING
    SHORT = "short"  # Section I
    COMPACT = "compact"  # doc_0_I


@dataclass
class Citation:
    """Represents a citation with validation"""
    reference_id: str  # Full reference: doc_0_Section_I_TITLE
    section_num: str  # Section number: I, 2.1, 0
    section_title: str  # Section title: GENERAL_UNDERTAKING
    contract_id: str  # Contract ID: doc_0
    confidence: float = 1.0  # Confidence score (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (concise format)"""
        return {
            "section_num": self.section_num,
            "section_title": self.section_title,
            "contract_id": self.contract_id,
            "confidence": self.confidence
        }

    @staticmethod
    def from_reference_string(ref: str) -> 'Citation':
        """Parse reference string into Citation object
        
        Expected format: {contract_id}_Section_{section_num}_{section_title}
        Example: doc_0_AGENCY_AGREEMENT_Section_II_DUTIES_OF_AGENT
        
        Note: contract_id can contain underscores, so we search for "_Section_" as delimiter
        """
        # Find the "_Section_" delimiter (most reliable anchor)
        section_marker = "_Section_"
        if section_marker not in ref:
            raise ValueError(f"Invalid reference format (missing '_Section_'): {ref}")
        
        # Split on "_Section_"
        contract_and_prefix, rest = ref.split(section_marker, 1)
        
        # Extract contract_id (everything before "_Section_")
        contract_id = contract_and_prefix
        
        # rest is "section_num_title", split on first underscore
        # But section_num might be empty (for numbered sections it's safe)
        if '_' not in rest:
            raise ValueError(f"Invalid reference format (malformed section part): {ref}")
        
        section_num, section_title = rest.split('_', 1)
        
        return Citation(
            reference_id=ref,
            section_num=section_num,
            section_title=section_title,
            contract_id=contract_id
        )


@dataclass
class StructuredOutput:
    """Structured output with citations, reasoning, and answer"""
    citations: List[Citation]
    reasoning: str
    answer: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (clean format)"""
        result = {
            "citations": [c.to_dict() for c in self.citations],
            "reasoning": self.reasoning,
            "answer": self.answer
        }
        # Only include metadata if it has content
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @staticmethod
    def from_dict(data: Dict) -> 'StructuredOutput':
        """Create StructuredOutput from dictionary (for deserialization)
        
        Handles multiple formats:
        1. Citations with reference_id: {"reference_id": "...", ...}
        2. Citations with components: {"section_num": "...", "section_title": "...", "contract_id": "..."}
        3. Citations as strings: "doc_0_Section_I_TITLE"
        """
        citations_raw = data.get("citations", [])
        citations = []
        
        for c in citations_raw:
            if isinstance(c, dict):
                # Check if we have reference_id (old format)
                if "reference_id" in c:
                    citations.append(Citation.from_reference_string(c["reference_id"]))
                # Otherwise reconstruct from components (new format)
                elif "contract_id" in c and "section_num" in c and "section_title" in c:
                    reference_id = f"{c['contract_id']}_Section_{c['section_num']}_{c['section_title']}"
                    citations.append(Citation.from_reference_string(reference_id))
            elif isinstance(c, str):
                citations.append(Citation.from_reference_string(c))
        
        return StructuredOutput(
            citations=citations,
            reasoning=data.get("reasoning", ""),
            answer=data.get("answer", "")
        )


# ==========================================
# REFERENCE EXTRACTION & VALIDATION
# ==========================================

class ReferenceExtractor:
    """Extract and validate reference identifiers from retrieved documents"""

    def __init__(self):
        """Initialize reference extractor"""
        self.reference_pattern = r'^(doc_\d+)_Section_([^_]+)_(.+)$'

    def extract_references(self, documents: List[Document]) -> List[Citation]:
        """
        Extract all valid reference identifiers from retrieved documents
        
        Args:
            documents: List of Document objects from hierarchical retrieval
                      (should have metadata: chunk_id, section_num, section_title, contract_id)
        
        Returns:
            List of Citation objects representing valid references
            
        Raises:
            ValueError: If document metadata is malformed
        """
        citations = []
        seen_refs = set()

        for doc in documents:
            # Extract reference components from metadata
            chunk_id = doc.metadata.get('chunk_id')
            section_num = doc.metadata.get('section_num')
            section_title = doc.metadata.get('section_title', '')
            contract_id = doc.metadata.get('contract_id')

            if not all([chunk_id, section_num, contract_id]):
                raise ValueError(f"Document missing required metadata: {doc.metadata}")

            # Build reference in standard format
            # Note: section_title may contain spaces, we preserve them
            ref = f"{contract_id}_Section_{section_num}_{section_title}"

            # Deduplicate
            if ref not in seen_refs:
                citation = Citation(
                    reference_id=ref,
                    section_num=section_num,
                    section_title=section_title,
                    contract_id=contract_id
                )
                citations.append(citation)
                seen_refs.add(ref)

        return citations

    def validate_reference(self, ref: str, valid_refs: List[str]) -> bool:
        """
        Validate if a reference exists in the valid reference set
        
        Args:
            ref: Reference string to validate
            valid_refs: List of valid reference strings
            
        Returns:
            True if reference is valid, False otherwise
        """
        return ref in valid_refs


# ==========================================
# GBNF GRAMMAR BUILDER
# ==========================================

class GBNFGrammarBuilder:
    """
    Build GBNF (GGML BNF) grammar for constrained generation
    
    GBNF is used with Outlines library for token-level constraint enforcement.
    This ensures LLM can only generate citations from the retrieved set.
    """

    def __init__(self, reference_format_template: str = 'doc_{contract_id}_Section_{section_num}_{section_title}'):
        """
        Initialize grammar builder
        
        Args:
            reference_format_template: Template for reference format
        """
        self.reference_format_template = reference_format_template

    def build_citation_rule(self, citations: List[Citation]) -> str:
        """
        Build GBNF rule for valid citations
        
        Args:
            citations: List of valid Citation objects
            
        Returns:
            GBNF rule string for citations
            
        Example output:
            citation ::= ( "doc_0_Section_I_GENERAL_UNDERTAKING" | "doc_0_Section_II_DUTIES" )
        """
        if not citations:
            return 'citation ::= ""'

        # Escape and quote each reference
        quoted_refs = [f'"{c.reference_id}"' for c in citations]
        
        # Build alternation rule
        citation_rule = 'citation ::= ( ' + ' | '.join(quoted_refs) + ' )'
        return citation_rule

    def build_json_schema(self, citations: List[Citation]) -> Dict[str, Any]:
        """
        Build JSON schema for constrained generation
        
        Args:
            citations: List of valid Citation objects
            
        Returns:
            JSON schema dictionary
            
        This schema restricts:
        - citations: array with items from valid reference set
        - reasoning: string (unrestricted reasoning)
        - answer: string (unrestricted answer)
        """
        valid_refs = [c.reference_id for c in citations]

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": valid_refs  # Only allow these references
                    },
                    "description": "List of citations from retrieved documents"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning chain"
                },
                "answer": {
                    "type": "string",
                    "description": "Final answer to the user question"
                }
            },
            "required": ["citations", "reasoning", "answer"],
            "additionalProperties": False
        }

        return schema

    def build_gbnf_grammar(self, citations: List[Citation]) -> str:
        """
        Build complete GBNF grammar for structured JSON output with citations
        
        Args:
            citations: List of valid Citation objects
            
        Returns:
            Complete GBNF grammar string for Outlines
            
        Note:
            GBNF (GGML BNF) is a constraint language for token generation.
            This builds a grammar that enforces the JSON structure with citation constraints.
        """
        if not citations:
            # Fallback: allow any JSON structure (no citation constraint)
            return self._build_base_json_grammar()

        # Build citation alternation
        valid_refs = [c.reference_id for c in citations]
        quoted_refs = ' | '.join([f'"{ref}"' for ref in valid_refs])

        # Build complete grammar
        grammar = f'''
root   ::= "{{\\"citations\\":" citation-list ",\\"reasoning\\":" reasoning-string ",\\"answer\\":" answer-string "}}"

citation-list ::= "[]" | "[" citation-item ("," citation-item)* "]"
citation-item ::= {quoted_refs}

reasoning-string ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""

answer-string ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""
'''
        return grammar.strip()

    def _build_base_json_grammar(self) -> str:
        """Fallback: Basic JSON grammar without citation constraints"""
        grammar = '''
root   ::= "{{\\"citations\\":" citation-list ",\\"reasoning\\":" reasoning-string ",\\"answer\\":" answer-string "}}"

citation-list ::= "[]" | "[" citation-string ("," citation-string)* "]"
citation-string ::= "\\"[^"]*\\""

reasoning-string ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""

answer-string ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""
'''
        return grammar.strip()


# ==========================================
# STRUCTURED GENERATION
# ==========================================

class RDGConstrainer:
    """
    Applies Retrieval-Dependent Grammar constraints to generation
    
    Implements token-level masking to ensure citations stay within valid set.
    Can use Outlines (recommended) or custom logit masking (fallback).
    """

    def __init__(self, grammar_builder: Optional[GBNFGrammarBuilder] = None):
        """
        Initialize RDG constrainer
        
        Args:
            grammar_builder: GBNFGrammarBuilder instance (optional, will create if None)
        """
        self.grammar_builder = grammar_builder or GBNFGrammarBuilder()
        self._outlines_available = self._check_outlines_availability()

    def _check_outlines_availability(self) -> bool:
        """Check if Outlines library is available"""
        try:
            import outlines
            return True
        except ImportError:
            return False

    def build_constraints(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Build constraints for a given set of retrieved documents
        
        Args:
            documents: Retrieved documents from hierarchical retrieval
            
        Returns:
            Dictionary with:
            - valid_citations: List of Citation objects
            - json_schema: JSON schema for validation
            - gbnf_grammar: GBNF grammar string (if Outlines available)
            - valid_reference_strings: List of reference strings for validation
        """
        extractor = ReferenceExtractor()
        citations = extractor.extract_references(documents)

        constraints = {
            'valid_citations': citations,
            'json_schema': self.grammar_builder.build_json_schema(citations),
            'valid_reference_strings': [c.reference_id for c in citations],
        }

        # Build GBNF grammar only if Outlines is available (optional for dev phase)
        if self._outlines_available:
            constraints['gbnf_grammar'] = self.grammar_builder.build_gbnf_grammar(citations)

        return constraints

    def get_generation_prompt(
        self,
        question: str,
        context: str,
        use_instruction: bool = True
    ) -> str:
        """
        Generate prompt for constrained generation
        
        Args:
            question: User question
            context: Retrieved context with references
            use_instruction: Whether to include JSON instruction
            
        Returns:
            Formatted prompt string
        """
        if use_instruction:
            instruction = (
                "Provide your response in the following JSON format:\n"
                "{\n"
                '  "citations": [<list of citations from the context above>],\n'
                '  "reasoning": "<step-by-step reasoning>",\n'
                '  "answer": "<final answer>"\n'
                "}\n\n"
                "Important: Only cite sources from the context provided above. "
                "Do NOT fabricate citations."
            )
        else:
            instruction = ""

        prompt = (
            f"{instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Response:"
        )
        return prompt.strip()


# ==========================================
# CITATION VERIFICATION & VALIDATION
# ==========================================

class CitationValidator:
    """Validate generated citations against retrieved documents"""

    def __init__(self, valid_references: List[str]):
        """
        Initialize validator with valid references
        
        Args:
            valid_references: List of valid reference strings from retrieval
        """
        self.valid_references = set(valid_references)

    def validate_output(self, output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate generated output against constraints
        
        Args:
            output: Generated output (should be JSON-like dict)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        if 'citations' not in output:
            errors.append("Missing 'citations' field")
        if 'reasoning' not in output:
            errors.append("Missing 'reasoning' field")
        if 'answer' not in output:
            errors.append("Missing 'answer' field")

        if errors:
            return False, errors

        # Validate citations
        citations = output.get('citations', [])
        if not isinstance(citations, list):
            errors.append("'citations' must be a list")
            return False, errors

        invalid_citations = []
        for citation in citations:
            if citation not in self.valid_references:
                invalid_citations.append(citation)

        if invalid_citations:
            errors.append(
                f"Invalid citations (not in retrieved set): {invalid_citations}"
            )

        return len(errors) == 0, errors

    def fix_citations(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to fix invalid citations by removing them
        (Conservative approach: better to remove than fabricate)
        
        Args:
            output: Generated output with potentially invalid citations
            
        Returns:
            Fixed output with only valid citations
        """
        citations = output.get('citations', [])
        valid_citations = [c for c in citations if c in self.valid_references]

        fixed_output = output.copy()
        fixed_output['citations'] = valid_citations

        return fixed_output


# ==========================================
# END-TO-END RDG PIPELINE
# ==========================================

class RDGPipeline:
    """
    End-to-end pipeline for retrieval-dependent generation
    
    Workflow:
    1. Extract valid citations from retrieved documents
    2. Build grammar constraints
    3. Generate prompt with constraints
    4. Generate structured output (via LLM)
    5. Validate and fix citations
    """

    def __init__(self, use_outlines: bool = False):
        """
        Initialize RDG pipeline
        
        Args:
            use_outlines: Whether to use Outlines for token-level constraints
                         (requires `pip install outlines`)
        """
        self.grammar_builder = GBNFGrammarBuilder()
        self.constrainer = RDGConstrainer(self.grammar_builder)
        self.use_outlines = use_outlines and self.constrainer._outlines_available

    def prepare_generation(self, documents: List[Document], question: str, context_text: str) -> Dict[str, Any]:
        """
        Prepare generation with constraints
        
        Args:
            documents: Retrieved documents
            question: User question
            context_text: Formatted context text
            
        Returns:
            Dictionary with:
            - prompt: Generation prompt
            - constraints: Constraint dictionary
            - valid_references: List of valid reference strings
        """
        constraints = self.constrainer.build_constraints(documents)
        prompt = self.constrainer.get_generation_prompt(question, context_text)

        return {
            'prompt': prompt,
            'constraints': constraints,
            'valid_references': constraints['valid_reference_strings']
        }

    def validate_generation(self, raw_output: str, valid_references: List[str]) -> Tuple[StructuredOutput, List[str]]:
        """
        Validate and parse generated output
        
        Args:
            raw_output: Raw text output from LLM
            valid_references: List of valid reference strings
            
        Returns:
            Tuple of (StructuredOutput, validation_errors)
        """
        # Try to parse JSON
        try:
            output_dict = json.loads(raw_output)
        except json.JSONDecodeError as e:
            return None, [f"Invalid JSON: {str(e)}"]

        # Validate citations
        validator = CitationValidator(valid_references)
        is_valid, errors = validator.validate_output(output_dict)

        if not is_valid:
            # Try to fix by removing invalid citations
            output_dict = validator.fix_citations(output_dict)

        # Convert to StructuredOutput
        try:
            structured = StructuredOutput.from_dict(output_dict)
            return structured, errors
        except Exception as e:
            return None, [f"Failed to construct StructuredOutput: {str(e)}"]


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def build_json_schema(citations: List[Citation]) -> Dict[str, Any]:
    """
    Convenience function to build JSON schema
    
    Args:
        citations: List of valid citations
        
    Returns:
        JSON schema dictionary
    """
    builder = GBNFGrammarBuilder()
    return builder.build_json_schema(citations)


def extract_citations_from_docs(documents: List[Document]) -> List[Citation]:
    """
    Convenience function to extract citations from documents
    
    Args:
        documents: Retrieved documents
        
    Returns:
        List of Citation objects
    """
    extractor = ReferenceExtractor()
    return extractor.extract_references(documents)


# End of module
