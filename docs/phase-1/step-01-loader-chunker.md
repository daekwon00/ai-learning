# Phase 1 - Step 01: 문서 로더 + 텍스트 청킹 (완료)

## 목표
- PyPDFLoader/TextLoader로 문서를 로드한다
- RecursiveCharacterTextSplitter로 500자/100자 오버랩 청킹을 구현한다

## 작업 항목
- [x] `langchain-rag/modules/loader.py` — 문서 로더 구현
  - PyPDFLoader (PDF), TextLoader (TXT) 지원
  - 디렉토리 내 파일 자동 탐지
- [x] `langchain-rag/modules/chunker.py` — 텍스트 청킹 구현
  - RecursiveCharacterTextSplitter (500자, 100자 오버랩)
  - 한국어 경계 분리자: `["다.\n", "다. ", ".\n", ". "]`
- [x] 각 모듈 독립 실행 테스트 (`if __name__ == "__main__"`)

## 완료 조건
- [x] `python modules/loader.py ./data`로 문서가 로드된다
- [x] `python modules/chunker.py`로 청킹이 수행된다

## 참고
- 파일: `langchain-rag/modules/loader.py`, `chunker.py`
- rag-project의 직접 구현 → LangChain 추상화로 전환
