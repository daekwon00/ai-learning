# Phase 2 - Step 01: 커스텀 도구 개발

## 목표
- LangChain Tool 인터페이스를 이해하고 커스텀 도구를 구현한다
- yfinance 기반 주가 조회 도구와 계산기 도구를 만든다

## 작업 항목
- [ ] `ai-agent/requirements.txt` 작성 및 패키지 설치
- [ ] `ai-agent/tools/stock_tool.py` — yfinance 주가 조회 도구 구현
  - 종목 코드로 현재가, 변동률, 기본 정보 조회
  - `@tool` 데코레이터 또는 `BaseTool` 상속
- [ ] `ai-agent/tools/calculator_tool.py` — 계산기 도구 구현
  - 수학 표현식 평가 (안전한 방식)
- [ ] `ai-agent/tools/news_tool.py` — 뉴스 검색 도구 구현 (간단 버전)
- [ ] 각 도구 독립 실행 테스트 (`if __name__ == "__main__"`)

## 완료 조건
- 3개 도구가 각각 독립 실행되어 올바른 결과를 반환한다
- `from tools.stock_tool import stock_tool` 등으로 import 가능하다

## 참고
- LangChain `@tool` 데코레이터: `from langchain_core.tools import tool`
- yfinance: `pip install yfinance`
- Phase 0의 `stock_analysis.ipynb`에서 yfinance 사용 경험 활용
