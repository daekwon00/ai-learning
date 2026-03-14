# Phase 4 - Step 01: FastAPI 서버 + RAG 엔드포인트

## 목표
- FastAPI 서버를 구성하고 Phase 1의 RAG를 API로 제공한다

## 작업 항목
- [ ] `deployment/requirements.txt` 작성 및 패키지 설치
- [ ] `deployment/app/main.py` — FastAPI 앱 엔트리포인트
- [ ] `deployment/app/services/rag_service.py` — RAG 파이프라인 서비스
  - langchain-rag 모듈 재사용
- [ ] `deployment/app/routers/chat.py` — `/api/chat` 엔드포인트
  - POST: 질문 → RAG 응답 + 출처
- [ ] `.env.example` 작성
- [ ] Swagger UI (`/docs`)에서 테스트

## 완료 조건
- `uvicorn app.main:app`으로 서버가 실행된다
- `/api/chat`에 질문을 보내면 RAG 응답이 반환된다
- `/docs`에서 API를 테스트할 수 있다

## 참고
- `from fastapi import FastAPI, APIRouter`
- langchain-rag의 모듈을 서비스 레이어에서 import하여 재사용
