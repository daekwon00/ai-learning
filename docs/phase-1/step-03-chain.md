# Phase 1 - Step 03: RAG 체인 + 대화형 RAG (완료)

## 목표
- LangChain create_retrieval_chain으로 QA 체인을 구성한다
- history_aware_retriever로 대화형 RAG를 구현한다

## 작업 항목
- [x] `langchain-rag/modules/chain.py` — QA 체인 구현
  - `create_retrieval_chain` + `create_stuff_documents_chain`
  - GPT-4o (langchain_openai.ChatOpenAI)
  - 시스템 프롬프트 설정 (문서 기반 QA)
- [x] 출처 표시 기능 (source_documents 반환)
- [x] `create_history_aware_retriever`로 대화 이력 지원
  - 이전 대화 컨텍스트를 반영한 검색 쿼리 재구성
- [x] 독립 실행 테스트

## 완료 조건
- [x] 질문에 대해 RAG 응답 + 출처가 반환된다
- [x] 멀티턴 대화에서 컨텍스트가 유지된다

## 참고
- 파일: `langchain-rag/modules/chain.py`
- `ConversationalRetrievalChain`은 deprecated → `create_history_aware_retriever` + `create_retrieval_chain` 조합
- LangChain 1.x: `from langchain_classic.chains import create_retrieval_chain`
