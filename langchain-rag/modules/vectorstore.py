"""벡터 스토어 모듈

rag-project 대응: lib/ai/embedding.ts + lib/db/index.ts
- 임베딩 생성: HuggingFace (기본, 무료/로컬) / OpenAI (비교용)
- 벡터 DB: ChromaDB (rag-project의 Supabase pgvector 대응)
- 유사도 검색: similarity_search
"""

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

# ChromaDB 저장 경로
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "rag_documents"


def get_embedding_model(provider: str = "huggingface") -> Embeddings:
    """임베딩 모델을 반환한다.

    Args:
        provider: "huggingface" (기본, 무료/로컬) 또는 "openai"

    Returns:
        Embeddings 인스턴스
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    # 기본: HuggingFace all-MiniLM-L6-v2 (무료, ~90MB)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "mps"},  # M4 Mac MPS 가속
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vectorstore(
    documents: list[Document],
    provider: str = "huggingface",
    persist_directory: str = CHROMA_DIR,
) -> Chroma:
    """문서를 임베딩하여 ChromaDB에 저장한다.

    Args:
        documents: 청크된 Document 리스트
        provider: 임베딩 모델 프로바이더
        persist_directory: ChromaDB 저장 경로

    Returns:
        Chroma 인스턴스
    """
    embedding_model = get_embedding_model(provider)

    print(f"벡터 스토어 생성 중... (provider={provider}, 문서 {len(documents)}개)")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )
    print(f"벡터 스토어 생성 완료: {persist_directory}")
    return vectorstore


def get_vectorstore(
    provider: str = "huggingface",
    persist_directory: str = CHROMA_DIR,
) -> Chroma:
    """기존 ChromaDB를 로드한다.

    Args:
        provider: 임베딩 모델 프로바이더 (생성 시와 동일해야 함)
        persist_directory: ChromaDB 저장 경로

    Returns:
        Chroma 인스턴스
    """
    embedding_model = get_embedding_model(provider)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )


def similarity_search(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
) -> list[Document]:
    """유사도 검색을 수행한다.

    Args:
        vectorstore: Chroma 인스턴스
        query: 검색 쿼리
        k: 반환할 문서 수

    Returns:
        유사도 높은 순서의 Document 리스트
    """
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\n검색 쿼리: '{query}'")
    print(f"상위 {len(results)}개 결과:")
    for i, (doc, score) in enumerate(results):
        print(f"  [{i+1}] score={score:.4f} | source={doc.metadata.get('source', 'N/A')} | {doc.page_content[:80]}...")

    return [doc for doc, _ in results]


if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)

    # 벡터 스토어 생성
    vs = create_vectorstore(chunks)

    # 유사도 검색 테스트
    similarity_search(vs, "RAG란 무엇인가?")
    similarity_search(vs, "Next.js의 주요 특징")
