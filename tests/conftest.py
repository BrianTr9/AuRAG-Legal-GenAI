"""
Pytest fixtures for chunking tests
Provides: chunker, contract_text, parents, children
"""

import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chunking import HierarchicalChunker, Chunk


# Sample contract text with numbered sections
SAMPLE_CONTRACT = """
1. PROVISION OF SERVICES.

(a) Provider agrees to provide the Services set forth in Exhibit A ("Services").

(b) The Services shall be provided in a professional and timely manner, in accordance with industry standards and best practices.

(c) Provider represents and warrants that it has the necessary expertise and experience to perform the Services.

2. RESPONSIBILITY FOR WAGES AND FEES.

(a) For such time as any employees of Provider are provided to perform the Services, Provider shall be responsible for all compensation, benefits, taxes, insurance, and other costs associated with such employees.

(b) Recipient shall not be responsible for any employment-related matters with respect to Provider's employees.

(c) Provider shall indemnify Recipient for any claims arising from Provider's failure to meet its employment obligations.

3. TERMINATION OF AGREEMENT.

(a) Either party may terminate this Agreement at any time without cause upon ninety (90) days' prior written notice to the other party.

(b) Upon termination, all outstanding invoices shall be paid within thirty (30) days.

(c) Provider shall return all Recipient materials and data within ten (10) days of termination.

4. CONFIDENTIALITY.

(a) Each party shall maintain the confidentiality of any proprietary information disclosed by the other party.

(b) Confidential information shall not be disclosed to third parties without prior written consent.

(c) The receiving party shall implement reasonable security measures to protect confidential information.

(d) These confidentiality obligations shall survive termination of this Agreement for a period of two (2) years.

5. INDEMNIFICATION.

(a) Provider shall indemnify, defend, and hold harmless Recipient from any claims, damages, or losses arising from Provider's breach of this Agreement or negligence in performing the Services.

(b) The indemnification obligation shall apply to all third-party claims, including attorney's fees and court costs.

6. ENTIRE AGREEMENT.

This Agreement, together with all exhibits and schedules, constitutes the entire agreement between the parties and supersedes all prior negotiations and agreements.
"""


@pytest.fixture
def chunker():
    """Fixture: HierarchicalChunker instance"""
    return HierarchicalChunker(
        child_size=300,
        child_overlap=90,
        encoding_name="cl100k_base",
        semantic_boundary_tolerance=30
    )


@pytest.fixture
def contract_text():
    """Fixture: Sample contract text"""
    return SAMPLE_CONTRACT


@pytest.fixture
def parents(chunker, contract_text):
    """Fixture: Parent chunks"""
    sections = chunker.extract_sections(contract_text)
    parents = chunker.create_parent_chunks(sections, contract_id="test_contract_001")
    return parents


@pytest.fixture
def children(chunker, parents):
    """Fixture: Child chunks"""
    children = []
    for parent in parents:
        child_chunks = chunker.create_child_chunks(parent)
        children.extend(child_chunks)
    return children
