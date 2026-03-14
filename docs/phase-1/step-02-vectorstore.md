# Phase 1 - Step 02: 벡터 스토어 (ChromaDB + HuggingFace) (완료)

## 목표
- ChromaDB에 문서 임베딩을 저장하고 유사도 검색을 수행한다
- HuggingFace 로컬 임베딩 (all-MiniLM-L6-v2)을 활용한다

## 작업 항목
- [x] `langchain-rag/modules/vectorstore.py` — 벡터 스토어 구현
  - `langchain_chroma.Chroma` 사용 (구 `langchain.vectorstores.Chroma` 아님)
  - `langchain_huggingface.HuggingFaceEmbeddings` (all-MiniLM-L6-v2)
  - M4 Mac `device="mps"` 가속 적용
- [x] 문서 임베딩 생성 및 ChromaDB 저장
- [x] 유사도 검색 테스트
- [x] 독립 실행 테스트

## 완료 조건
- [x] ChromaDB에 문서가 저장되고 유사도 검색이 동작한다
- [x] `python modules/vectorstore.py`로 검색 결과가 반환된다

## 참고
- 파일: `langchain-rag/modules/vectorstore.py`
- rag-project: OpenAI 임베딩($) + pgvector → 여기서는 HuggingFace 로컬(무료) + ChromaDB
