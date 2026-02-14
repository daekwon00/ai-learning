"""텍스트 청킹 모듈

rag-project 대응: lib/chunker.ts
- chunk_size=500, overlap=100 (동일)
- 한국어 문장 경계 separator 포함
- RecursiveCharacterTextSplitter 사용
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# rag-project의 chunkText와 동일한 파라미터
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 한국어 + 영어 문장 경계를 포함하는 separator 리스트
# RecursiveCharacterTextSplitter는 순서대로 시도하며, 가장 먼저 매칭되는 것으로 분할
SEPARATORS = [
    "\n\n",    # 문단 경계 (최우선)
    "다.\n",   # 한국어 문장 끝 + 줄바꿈
    "다. ",    # 한국어 문장 끝 + 공백
    ".\n",     # 영어 문장 끝 + 줄바꿈
    ". ",      # 영어 문장 끝 + 공백
    "\n",      # 줄바꿈
    " ",       # 공백
    "",        # 문자 단위 (최후 수단)
]


def get_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """RecursiveCharacterTextSplitter 인스턴스를 반환한다."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """문서 리스트를 청킹한다.

    Args:
        documents: Document 리스트
        chunk_size: 청크 크기 (기본 500)
        chunk_overlap: 오버랩 크기 (기본 100)

    Returns:
        청크된 Document 리스트 (원본 메타데이터 유지)
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    print(f"청킹 완료: {len(documents)}개 문서 → {len(chunks)}개 청크")
    print(f"  청크 크기 범위: {min(len(c.page_content) for c in chunks)}~{max(len(c.page_content) for c in chunks)}자")

    return chunks


if __name__ == "__main__":
    from loader import load_documents
    import os

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)

    print("\n--- 청크 샘플 (처음 3개) ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[청크 {i+1}] source={chunk.metadata['source']}, 길이={len(chunk.page_content)}자")
        print(chunk.page_content[:200] + "...")
