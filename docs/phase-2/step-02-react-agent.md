# Phase 2 - Step 02: ReAct Agent 구현

## 목표
- ReAct (Reasoning + Acting) 패턴을 이해하고 Agent를 구현한다
- Step 01에서 만든 도구를 Agent에 연결한다

## 작업 항목
- [ ] `ai-agent/agents/react_agent.py` — ReAct Agent 구현
  - `create_react_agent` 또는 `AgentExecutor` 사용
  - stock_tool, calculator_tool, news_tool 연결
- [ ] ReAct 프롬프트 템플릿 설정
- [ ] 단일 턴 테스트: "삼성전자 현재 주가 알려줘"
- [ ] Agent의 사고 과정(Thought → Action → Observation) 로깅 확인

## 완료 조건
- Agent가 질문에 따라 적절한 도구를 선택하고 실행한다
- Agent의 추론 과정이 출력된다

## 참고
- `from langchain.agents import create_react_agent, AgentExecutor`
- `from langchain_openai import ChatOpenAI`
- OpenAI API 키: `ai-agent/.env` 또는 `langchain-rag/.env` 공유
