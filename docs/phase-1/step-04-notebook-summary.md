# Phase 1 - Step 04: 통합 노트북 + 정리 (완료)

## 목표
- 전체 RAG 파이프라인을 하나의 노트북에 통합한다
- TypeScript(rag-project) vs Python(langchain-rag) 비교를 정리한다

## 작업 항목
- [x] `langchain-rag/rag_pipeline.ipynb` — 통합 노트북 작성
  - 문서 로딩 → 청킹 → 벡터 저장 → QA 체인 → 대화형 RAG (5섹션)
- [x] `langchain-rag/summary.md` — TS vs Python 비교 정리
  - 파일 대응표, 핵심 코드 대응, 주요 차이점
- [x] `langchain-rag/README.md` 업데이트 (실행 방법, 디렉토리 구조)
- [x] `langchain-rag/requirements.txt` 작성
- [x] 노트북 실행 결과 포함 상태로 커밋

## 완료 조건
- [x] `rag_pipeline.ipynb`가 처음부터 끝까지 실행된다
- [x] summary.md에 비교 분석이 정리되어 있다
- [x] README.md에 실행 방법이 기록되어 있다

## 참고
- 파일: `langchain-rag/rag_pipeline.ipynb`, `summary.md`, `README.md`
- 학습 포인트: LangChain LCEL, HuggingFace 임베딩, ChromaDB, MPS 가속
