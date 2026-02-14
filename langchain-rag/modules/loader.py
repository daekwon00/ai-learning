"""문서 로더 모듈

rag-project 대응: app/api/ingest/route.ts
- PDF: PyPDFLoader (rag-project의 pdf-parse 대응)
- TXT: TextLoader (rag-project의 File.text() 대응)
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(directory: str) -> list[Document]:
    """디렉토리 내 모든 PDF/TXT 파일을 로드한다.

    Args:
        directory: 문서가 있는 디렉토리 경로

    Returns:
        Document 리스트 (source 메타데이터 포함)
    """
    documents = []
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"디렉토리가 존재하지 않습니다: {directory}")

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["file_type"] = "pdf"
            documents.extend(docs)
            print(f"  [PDF] {file_path.name}: {len(docs)}페이지 로드")

        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["file_type"] = "txt"
            documents.extend(docs)
            print(f"  [TXT] {file_path.name}: {len(docs)}개 문서 로드")

    print(f"\n총 {len(documents)}개 문서 로드 완료")
    return documents


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data")
    docs = load_documents(data_dir)
    for doc in docs:
        print(f"  - {doc.metadata['source']} ({doc.metadata['file_type']}): {len(doc.page_content)}자")
