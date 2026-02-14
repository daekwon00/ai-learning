# Phase 4: FastAPI + Docker 배포

> Phase 1~3 결과물을 FastAPI 서버 + Docker로 통합 배포

## 학습 목표

- FastAPI: `/api/chat` (RAG), `/api/agent` (Agent)
- Streamlit 프론트엔드 (채팅 + 파일 업로드)
- Dockerfile + docker-compose.yml
- (선택) MLflow 모델 추적

## 디렉토리 구조

```
deployment/
├── README.md
├── requirements.txt
├── app/
│   ├── main.py             # FastAPI 엔트리포인트
│   ├── routers/
│   │   ├── chat.py         # /api/chat (RAG)
│   │   └── agent.py        # /api/agent (Agent)
│   └── services/
│       ├── rag_service.py
│       └── agent_service.py
├── streamlit_app.py        # Streamlit UI
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 환경

```bash
pip install fastapi uvicorn streamlit docker
```

## 검증

```bash
docker-compose up
# http://localhost:8000/docs  → FastAPI Swagger
# http://localhost:8501       → Streamlit UI
```

## 상태: 미착수
