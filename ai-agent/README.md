# Phase 2: AI Agent - Tool Calling

> LangChain Agent로 커스텀 도구 활용 + ReAct 패턴 구현

## 학습 목표

- 커스텀 도구 개발: 주가 조회(yfinance), 뉴스 검색, 계산기
- ReAct Agent + OpenAI Functions Agent
- 멀티턴 시나리오 ("삼성전자 주가 분석해줘")
- (선택) LangGraph 상태 기반 Agent

## 디렉토리 구조

```
ai-agent/
├── README.md
├── requirements.txt
├── agent_demo.ipynb        # 메인 노트북
├── tools/
│   ├── stock_tool.py       # yfinance 주가 조회
│   ├── news_tool.py        # 뉴스 검색
│   └── calculator_tool.py  # 계산기
└── agents/
    ├── react_agent.py      # ReAct Agent
    └── functions_agent.py  # OpenAI Functions Agent
```

## 환경

```bash
conda activate ai-dev
pip install langchain langchain-openai yfinance
```

## 상태: 미착수
