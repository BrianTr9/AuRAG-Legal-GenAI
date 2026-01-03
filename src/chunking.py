"""
Hierarchical Chunking for AuRAG Layer 1
========================================

General hierarchical chunking with semantic boundary detection.
Supports structured documents (contracts, legal docs, technical specs).

Strategy:
- Parent: Full sections (no size limit, typically 1500-3000 tokens)
- Child: ~300 tokens with 90-token overlap (30%)
- Semantic boundaries: \n\n → . → \n → ; → , (preserves paragraphs/sentences)
- Metadata: Child → Parent mapping for retrieval expansion

Author: AuRAG Team
Date: January 2026
"""

import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("⚠️  tiktoken not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip3", "install", "tiktoken"])
    import tiktoken


@dataclass
class Chunk:
    """
    Represents a hierarchical chunk with metadata
    
    Attributes:
        text: The actual text content
        chunk_id: Unique identifier (e.g., "parent_2.1" or "parent_2.1_child_0")
        chunk_type: Either "parent" or "child"
        parent_id: Parent chunk ID (None for parents, required for children)
        section_num: Section number from contract (e.g., "2.1")
        section_title: Section title (e.g., "OVERVIEW")
        token_count: Number of tokens in this chunk
        char_start: Starting character position in original document
        char_end: Ending character position in original document
        contract_id: Identifier of source contract
    """
    text: str
    chunk_id: str
    chunk_type: str  # "parent" or "child"
    parent_id: str = None
    section_num: str = None
    section_title: str = None
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    contract_id: str = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class HierarchicalChunker:
    """
    General hierarchical chunking system with semantic boundary detection
    
    Design Philosophy:
    1. Parents = Full sections (complete context)
    2. Children = Small chunks (~300 tokens, precise retrieval)
    3. Semantic boundaries (paragraph/sentence/clause aware)
    4. Retrieval: Search children → expand to parents
    5. Context window: 8K tokens fits 1-2 parents + 4-5 children
    """
    
    def __init__(
        self,
        child_size: int = 300,
        child_overlap: int = 90,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize chunker
        
        Args:
            child_size: Target size for child chunks in tokens (default: 300)
            child_overlap: Overlap between consecutive children in tokens (default: 90 = 30%)
            encoding_name: Tiktoken encoding (default: cl100k_base for GPT-4/Gemma)
        """
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        
        # Separator priority (from strongest to weakest boundary)
        self.separators = [
            ('\n\n', 'paragraph'),      # Double newline (paragraph break)
            ('.\n', 'sentence_line'),   # Sentence at line end
            ('. ', 'sentence'),         # Sentence boundary
            ('\n', 'line'),             # Single line break
            ('; ', 'semicolon'),        # Semicolon (clause separator)
            (', ', 'comma'),            # Comma separator
            (' ', 'word')               # Word boundary (last resort)
        ]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def extract_sections(self, contract_text: str) -> List[Dict]:
        """
        Extract hierarchical sections from structured document
        
        Expected format: "1.1 Provision of Services." or "2. Compensation." in running text
        Strategy: Group subsections (1.1, 1.2) into major section (1)
        
        Returns:
            List of section dictionaries with metadata
        """
        sections = []
        
        # Pattern matches numbered section format:
        # "1. Services." or "2.1 Provision of Services."
        pattern = r'(\d+(?:\.\d+)?)\s+([A-Z][^.]+)\.\s'
        
        # Find all section markers
        matches = list(re.finditer(pattern, contract_text))
        
        if not matches:
            print("⚠️  No structured sections found, using entire document as one parent")
            return [{
                'section_num': '1',
                'title': 'Full Document',
                'text': contract_text,
                'char_start': 0,
                'char_end': len(contract_text),
                'token_count': self.count_tokens(contract_text)
            }]
        
        # Group subsections into major sections
        # E.g., 1.1, 1.2, 1.3 → Section 1
        major_sections = {}
        
        for i, match in enumerate(matches):
            section_num = match.group(1)
            section_title = match.group(2).strip()
            
            # Get major section number (first digit before dot)
            if '.' in section_num:
                major_num = section_num.split('.')[0]
            else:
                major_num = section_num
            
            # Initialize major section if first time seeing it
            if major_num not in major_sections:
                major_sections[major_num] = {
                    'section_num': major_num,
                    'title': section_title,  # Use first subsection's title
                    'start_pos': match.start(),
                    'first_match_idx': i
                }
        
        # Now determine end positions for each major section
        result_sections = []
        major_nums = sorted(major_sections.keys(), key=int)
        
        for idx, major_num in enumerate(major_nums):
            sec = major_sections[major_num]
            start_pos = sec['start_pos']
            
            # Find end: start of next major section or end of doc
            if idx < len(major_nums) - 1:
                next_major = major_sections[major_nums[idx + 1]]
                end_pos = next_major['start_pos']
            else:
                end_pos = len(contract_text)
            
            section_text = contract_text[start_pos:end_pos].strip()
            
            result_sections.append({
                'section_num': major_num,
                'title': sec['title'],
                'text': section_text,
                'char_start': start_pos,
                'char_end': end_pos,
                'token_count': self.count_tokens(section_text)
            })
        
        return result_sections
    
    def create_parent_chunks(
        self,
        sections: List[Dict],
        contract_id: str = "contract"
    ) -> List[Chunk]:
        """
        Create parent chunks from sections
        
        Strategy: Each section becomes ONE parent chunk (full context)
        No size limit - legal text requires complete context
        
        Args:
            sections: List of extracted sections
            contract_id: Identifier for the source contract
            
        Returns:
            List of parent Chunk objects
        """
        parents = []
        
        for sec in sections:
            parent = Chunk(
                text=sec['text'],
                chunk_id=f"parent_{sec['section_num']}",
                chunk_type="parent",
                parent_id=None,  # Parents have no parent
                section_num=sec['section_num'],
                section_title=sec['title'],
                token_count=sec['token_count'],
                char_start=sec['char_start'],
                char_end=sec['char_end'],
                contract_id=contract_id
            )
            parents.append(parent)
        
        return parents
    
    def split_by_separators(self, text: str, target_size: int) -> List[str]:
        """
        Split text using separator priority hierarchy
        
        Priority: \\n\\n → .\\s → \\n → ;\\s → ,\\s → space
        
        Args:
            text: Text to split
            target_size: Target size in tokens
            
        Returns:
            List of text chunks
        """
        if self.count_tokens(text) <= target_size:
            return [text]
        
        # Try each separator in priority order
        for separator, sep_name in self.separators:
            parts = text.split(separator)
            
            if len(parts) <= 1:
                # This separator doesn't exist, try next
                continue
            
            # Rebuild chunks respecting target size
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for i, part in enumerate(parts):
                # Add separator back (except for last part)
                if i < len(parts) - 1:
                    part_with_sep = part + separator
                else:
                    part_with_sep = part
                
                part_tokens = self.count_tokens(part_with_sep)
                
                # Check if adding this part exceeds target
                if current_tokens + part_tokens <= target_size or not current_chunk:
                    current_chunk.append(part_with_sep)
                    current_tokens += part_tokens
                else:
                    # Save current chunk and start new one
                    chunks.append(''.join(current_chunk))
                    current_chunk = [part_with_sep]
                    current_tokens = part_tokens
            
            # Save last chunk
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            # Check if splitting was successful
            max_chunk_size = max(self.count_tokens(c) for c in chunks)
            if max_chunk_size <= target_size * 1.2:  # Allow 20% tolerance
                return [c.strip() for c in chunks if c.strip()]
        
        # If all separators failed, force split by tokens
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), target_size):
            chunk_tokens = tokens[i:i + target_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        
        return chunks
    
    def find_semantic_boundary(
        self,
        tokens: list,
        target_pos: int,
        tolerance: int = 30
    ) -> int:
        """
        Find semantic boundary near target position
        
        Priority: paragraph → sentence → line → semicolon → comma
        
        Args:
            tokens: List of tokens
            target_pos: Target position to find boundary near
            tolerance: Search range ±tolerance from target
            
        Returns:
            Best boundary position, or target_pos if none found
        """
        # Search range
        start_search = max(0, target_pos - tolerance)
        end_search = min(len(tokens), target_pos + tolerance)
        
        if start_search >= end_search:
            return target_pos
        
        # Decode search window to text
        search_tokens = tokens[start_search:end_search]
        search_text = self.tokenizer.decode(search_tokens)
        
        # Define separators by priority
        separators = [
            '\n\n',   # Paragraph break (highest priority)
            '.\n',    # Sentence at line end
            '. ',     # Sentence with space
            '\n',     # Line break
            '; ',     # Semicolon with space
            ', ',     # Comma with space
        ]
        
        best_boundary = None
        best_priority = -1
        
        for priority, sep in enumerate(separators):
            # Find all occurrences of this separator
            positions = []
            offset = 0
            while True:
                pos = search_text.find(sep, offset)
                if pos == -1:
                    break
                # Position after separator
                positions.append(pos + len(sep))
                offset = pos + 1
            
            if not positions:
                continue
            
            # Find closest to middle of search window
            middle_offset = len(search_text) // 2
            closest_pos = min(positions, key=lambda p: abs(p - middle_offset))
            
            # Convert text position to token position
            # Decode from start to this position to count tokens
            text_before = search_text[:closest_pos]
            tokens_before = self.tokenizer.encode(text_before)
            boundary_token_pos = start_search + len(tokens_before)
            
            # Check if this is better than previous
            if best_boundary is None or priority < best_priority:
                best_boundary = boundary_token_pos
                best_priority = priority
                
                # If we found paragraph break, that's best possible
                if priority == 0:
                    break
        
        # Return best boundary or target if none found
        return best_boundary if best_boundary is not None else target_pos
    
    def create_child_chunks(
        self,
        parent: Chunk
    ) -> List[Chunk]:
        """
        Create child chunks from parent with sliding window overlap
        AND semantic boundary detection
        
        Strategy: 
        1. Target 300-token chunks with 90-token overlap
        2. Adjust boundaries to respect semantic markers
        3. Priority: paragraph → sentence → line → clause → comma
        
        Args:
            parent: Parent chunk to split
            
        Returns:
            List of child Chunk objects
        """
        # Tokenize entire parent text
        parent_tokens = self.tokenizer.encode(parent.text)
        
        if len(parent_tokens) == 0:
            return []
        
        # Calculate stride (step size between chunks)
        stride = self.child_size - self.child_overlap
        
        if stride <= 0:
            stride = self.child_size
        
        children = []
        chunk_idx = 0
        position = 0
        
        while position < len(parent_tokens):
            # Target end position
            target_end = min(position + self.child_size, len(parent_tokens))
            
            # Find semantic boundary near target
            # Only adjust if not at document end
            if target_end < len(parent_tokens):
                actual_end = self.find_semantic_boundary(
                    parent_tokens, 
                    target_end,
                    tolerance=30  # ±30 tokens = ±10% flexibility
                )
            else:
                actual_end = target_end
            
            # Extract chunk tokens
            chunk_tokens = parent_tokens[position:actual_end]
            
            # Decode to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create child chunk
            child = Chunk(
                text=chunk_text,
                chunk_id=f"{parent.chunk_id}_child_{chunk_idx}",
                chunk_type="child",
                parent_id=parent.chunk_id,
                section_num=parent.section_num,
                section_title=parent.section_title,
                token_count=len(chunk_tokens),
                char_start=parent.char_start,
                char_end=parent.char_end,
                contract_id=parent.contract_id
            )
            children.append(child)
            
            # Move forward by stride (creates overlap)
            # Next chunk starts at (actual_end - overlap)
            position = actual_end - self.child_overlap
            chunk_idx += 1
            
            # Stop if we've covered the entire text
            if actual_end >= len(parent_tokens):
                break
        
        return children
    
    def chunk_contract(
        self,
        contract_text: str,
        contract_id: str = "contract_001"
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Main chunking pipeline
        
        Args:
            contract_text: Full contract text
            contract_id: Identifier for this contract
            
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        print(f"\n{'='*60}")
        print(f"🏗️  Chunking contract: {contract_id}")
        print(f"{'='*60}")
        
        # Step 1: Extract sections
        sections = self.extract_sections(contract_text)
        print(f"📄 Extracted {len(sections)} sections")
        
        # Step 2: Create parent chunks (full sections)
        parents = self.create_parent_chunks(sections, contract_id)
        print(f"👨 Created {len(parents)} parent chunks")
        
        # Step 3: Create child chunks with overlap
        all_children = []
        for parent in parents:
            children = self.create_child_chunks(parent)
            all_children.extend(children)
        
        print(f"👶 Created {len(all_children)} child chunks")
        
        # Statistics
        if parents:
            parent_tokens = [p.token_count for p in parents]
            print(f"\n📊 Parent Statistics:")
            print(f"   Avg: {sum(parent_tokens)/len(parent_tokens):.0f} tokens")
            print(f"   Min: {min(parent_tokens)} tokens")
            print(f"   Max: {max(parent_tokens)} tokens")
        
        if all_children:
            child_tokens = [c.token_count for c in all_children]
            print(f"\n📊 Child Statistics:")
            print(f"   Avg: {sum(child_tokens)/len(child_tokens):.0f} tokens")
            print(f"   Min: {min(child_tokens)} tokens")
            print(f"   Max: {max(child_tokens)} tokens")
        
        return parents, all_children
    
    def save_chunks(
        self,
        parents: List[Chunk],
        children: List[Chunk],
        output_path: str
    ):
        """
        Save chunks to JSON file with full metadata
        
        Args:
            parents: List of parent chunks
            children: List of child chunks
            output_path: Path to save JSON file
        """
        data = {
            'metadata': {
                'child_size': self.child_size,
                'child_overlap': self.child_overlap,
                'num_parents': len(parents),
                'num_children': len(children)
            },
            'parents': [p.to_dict() for p in parents],
            'children': [c.to_dict() for c in children]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved chunks to: {output_path}")


# ==================== DEMO ====================

def demo():
    """Demonstration of hierarchical chunking"""
    
    # Load sample CUAD contract
    contract_path = Path("LegalBench-RAG/corpus/cuad/ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt")
    
    if not contract_path.exists():
        print(f"❌ Contract not found: {contract_path}")
        print("📁 Current directory:", Path.cwd())
        return
    
    with open(contract_path, 'r', encoding='utf-8') as f:
        contract_text = f.read()
    
    print("="*60)
    print("🎯 AuRAG Layer 1: Hierarchical Chunking Demo")
    print("="*60)
    print(f"\n📄 Contract: {contract_path.name}")
    print(f"📏 Size: {len(contract_text):,} characters")
    
    # Create chunker
    chunker = HierarchicalChunker(
        child_size=300,
        child_overlap=90  # 30% overlap
    )
    
    # Chunk the contract
    parents, children = chunker.chunk_contract(
        contract_text,
        contract_id=contract_path.stem
    )
    
    # Display samples
    print("\n" + "="*60)
    print("📋 Sample Parent Chunk")
    print("="*60)
    
    if parents:
        sample_parent = parents[0]
        print(f"ID: {sample_parent.chunk_id}")
        print(f"Section: {sample_parent.section_num} - {sample_parent.section_title}")
        print(f"Tokens: {sample_parent.token_count}")
        print(f"\nText preview (first 400 chars):")
        print("-" * 60)
        print(sample_parent.text[:400])
        print("...")
        
        # Show children of this parent
        parent_children = [c for c in children if c.parent_id == sample_parent.chunk_id]
        
        print("\n" + "="*60)
        print(f"📋 Child Chunks from Parent '{sample_parent.chunk_id}' ({len(parent_children)} total)")
        print("="*60)
        
        for i, child in enumerate(parent_children[:3]):  # Show first 3
            print(f"\n[Child {i+1}]")
            print(f"  ID: {child.chunk_id}")
            print(f"  Parent: {child.parent_id}")
            print(f"  Tokens: {child.token_count}")
            print(f"  Text: {child.text[:200]}...")
    
    # Save to JSON
    output_path = "cuad_chunks_output.json"
    chunker.save_chunks(parents, children, output_path)
    
    print("\n" + "="*60)
    print("✅ Demo Complete!")
    print("="*60)
    print(f"\n🎯 Next steps:")
    print(f"  1. Index children in vector DB (Qdrant/ChromaDB)")
    print(f"  2. Store parent_id in metadata")
    print(f"  3. Retrieve top-K children → deduplicate to parents")
    print(f"  4. Pass parents to LLM (Gemma-2-9B-It)")


if __name__ == "__main__":
    demo()
