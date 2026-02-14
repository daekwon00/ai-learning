# AI Learning Project

> Python/ML 기초부터 LangChain RAG, AI Agent, BERT Fine-tuning, 배포까지

## 프로젝트 개요

Phase 기반 AI/ML 학습 프로젝트. 별도 rag-project(Next.js + Supabase pgvector)에서 RAG 핵심을 이미 구현했으며, 여기서는 **Python 생태계** 중심으로 학습합니다.

### rag-project에서 이미 커버한 것

- RAG 핵심 (청킹, 임베딩, 벡터검색, 출처추적) — pgvector 전체 구현
- OpenAI API (GPT-4o, 임베딩) — 실전 사용
- 스트리밍 응답 — Vercel AI SDK
- PDF 문서 처리, 채팅 UI, 프로덕션 배포 (Vercel)

## 디렉토리 구조

```
ai-learning/
├── python-ml-basics/     ← 완료: Python/ML 기초 (NumPy, Pandas, Scikit-learn, PyTorch)
├── langchain-rag/        ← Phase 1: Python LangChain + ChromaDB RAG
├── ai-agent/             ← Phase 2: LangChain Agent (Tool Calling, ReAct)
├── pytorch-bert/         ← Phase 3: BERT Fine-tuning (금융 뉴스 감성 분석)
└── deployment/           ← Phase 4: FastAPI + Docker 배포
```

## Phase별 진행 상황

| Phase | 주제 | 기간 | 상태 |
|-------|------|------|------|
| - | Python/ML 기초 | 완료 | :white_check_mark: |
| 1 | Python LangChain RAG + ChromaDB | 2~3일 | 미착수 |
| 2 | AI Agent (Tool Calling, ReAct) | 2~3일 | 미착수 |
| 3 | PyTorch BERT Fine-tuning | 2~3일 | 미착수 |
| 4 | FastAPI + Docker 배포 | 2~3일 | 미착수 |

## 시작하기

```bash
conda activate ai-dev
```

각 Phase 디렉토리의 README.md에 상세 안내가 있습니다.

## 환경

- **Python 3.11** (Conda: ai-dev)
- 주요 패키지: LangChain, ChromaDB, PyTorch, Transformers, FastAPI

## 작성자

**YDK** | 2026.01 ~
