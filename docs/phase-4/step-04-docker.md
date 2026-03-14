# Phase 4 - Step 04: Docker 컨테이너화 + 배포

## 목표
- 전체 시스템을 Docker로 컨테이너화한다

## 작업 항목
- [ ] `deployment/Dockerfile` — FastAPI 서버 이미지
- [ ] `deployment/docker-compose.yml` — 멀티 서비스 구성
  - FastAPI (port 8000)
  - Streamlit (port 8501)
- [ ] 환경 변수 관리 (.env → Docker)
- [ ] `docker-compose up`으로 전체 시스템 실행 테스트
- [ ] `deployment/summary.md` — Phase 4 학습 요약
- [ ] `deployment/README.md` 업데이트 (실행 방법, 아키텍처 다이어그램)

## 완료 조건
- `docker-compose up`으로 FastAPI + Streamlit이 동시에 실행된다
- http://localhost:8000/docs → FastAPI Swagger
- http://localhost:8501 → Streamlit UI
- 두 서비스가 연동되어 정상 동작한다

## 참고
- 최종 검증 단계 — 전체 포트폴리오 완성
- (선택) MLflow 모델 추적 컨테이너 추가
