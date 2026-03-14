# Phase 2 - Step 03: OpenAI Functions Agent + 멀티턴

## 목표
- OpenAI Functions 기반 Agent를 구현한다
- 멀티턴 대화 시나리오를 구성한다

## 작업 항목
- [ ] `ai-agent/agents/functions_agent.py` — OpenAI Functions Agent 구현
  - `create_openai_functions_agent` 또는 최신 `create_tool_calling_agent` 사용
- [ ] 멀티턴 대화 지원 (ChatMessageHistory 활용)
- [ ] 복합 시나리오 테스트:
  - "삼성전자 주가 분석해줘" → 주가 조회 + 분석 코멘트
  - "그 회사의 최근 뉴스도 찾아줘" → 컨텍스트 유지 + 뉴스 검색
- [ ] ReAct Agent vs Functions Agent 비교 정리

## 완료 조건
- Functions Agent가 멀티턴 대화에서 컨텍스트를 유지한다
- 두 Agent 방식의 차이점을 설명할 수 있다

## 참고
- `create_tool_calling_agent`가 `create_openai_functions_agent`의 최신 대체
- `from langchain_core.chat_history import InMemoryChatMessageHistory`
