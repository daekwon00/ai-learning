# AI Learning Project - Product Requirements Document

## 개요
Phase 기반 AI/ML 학습 포트폴리오 프로젝트. 별도 rag-project(Next.js + Supabase pgvector)에서 RAG를 이미 구현했으며, 여기서는 **Python 생태계** 중심으로 전환 학습한다.

## 목표
- Python 기반 AI/ML 개발 역량 구축 (PyTorch, BERT)
- LLM 플랫폼 실무 경험 (OpenAI API, LangChain)
- RAG 시스템 구축 및 AI Agent 개발
- MLOps 기초 및 포트폴리오 완성
- 최종 목표: Enterprise AI System Developer

## 기술 스택
- Python 3.11 (Conda: ai-dev), M4 Mac
- LangChain 1.x, ChromaDB, HuggingFace Embeddings
- OpenAI GPT-4o
- PyTorch, HuggingFace Transformers
- FastAPI, Streamlit, Docker

## 기능 요구사항

### Phase 0: Python/ML 기초 (완료)
- 금융 데이터 수집/분석 (yfinance, Pandas)
- 신용평가 ML 파이프라인 (Scikit-learn)
- PyTorch 기초 + MNIST CNN

### Phase 1: Python LangChain RAG (완료)
- 모듈식 RAG 파이프라인 (loader → chunker → vectorstore → chain)
- PyPDFLoader + RecursiveCharacterTextSplitter (500자/100자 오버랩)
- ChromaDB + HuggingFace 임베딩 (all-MiniLM-L6-v2)
- 대화형 RAG (create_history_aware_retriever + create_retrieval_chain)
- 출처 표시 포함

### Phase 2: AI Agent - Tool Calling
- 커스텀 도구 개발: 주가 조회(yfinance), 뉴스 검색, 계산기
- ReAct Agent + OpenAI Functions Agent
- 멀티턴 시나리오 ("삼성전자 주가 분석해줘")
- (선택) LangGraph 상태 기반 Agent

### Phase 3: PyTorch + HuggingFace BERT Fine-tuning
- BERT Fine-tuning 금융 뉴스 감성 분석 (3-class: positive/negative/neutral)
- HuggingFace Trainer API
- Financial PhraseBank 또는 한국어 금융 뉴스 데이터셋
- 평가: Accuracy, F1, Confusion Matrix

### Phase 4: FastAPI + Docker 배포
- FastAPI: `/api/chat` (RAG), `/api/agent` (Agent)
- Streamlit 프론트엔드 (채팅 + 파일 업로드)
- Dockerfile + docker-compose.yml
- (선택) MLflow 모델 추적

## 비기능 요구사항
- 각 Phase는 독립적으로 실행 가능해야 함
- 노트북은 실행 결과 포함 상태로 커밋
- 각 Phase에 README.md, summary.md, requirements.txt 포함
- M4 Mac MPS 가속 활용 (HuggingFace, PyTorch)

## 검증 방법
- 각 Phase 완료 시: Notebook 실행 결과 확인 + GitHub 커밋
- Phase 4 완료 시: `docker-compose up`으로 전체 시스템 동작 확인

## 작성자
**YDK** | 2026.01 ~
