# Phase 4 - Step 02: Agent 엔드포인트 + 서비스 통합

## 목표
- Phase 2의 Agent를 API로 제공한다
- RAG + Agent 서비스를 통합한다

## 작업 항목
- [ ] `deployment/app/services/agent_service.py` — Agent 서비스
  - ai-agent 모듈 재사용
- [ ] `deployment/app/routers/agent.py` — `/api/agent` 엔드포인트
  - POST: 질문 → Agent 응답 (도구 사용 포함)
- [ ] 헬스체크 엔드포인트 (`/health`)
- [ ] 에러 핸들링 및 타임아웃 설정
- [ ] 두 엔드포인트 통합 테스트

## 완료 조건
- `/api/agent`에 질문을 보내면 Agent 응답이 반환된다
- `/api/chat`과 `/api/agent` 모두 정상 동작한다

## 참고
- Agent 응답은 시간이 걸릴 수 있으므로 적절한 타임아웃 설정
