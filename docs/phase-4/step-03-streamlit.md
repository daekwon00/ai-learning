# Phase 4 - Step 03: Streamlit 프론트엔드

## 목표
- Streamlit으로 채팅 UI를 구현한다

## 작업 항목
- [ ] `deployment/streamlit_app.py` — Streamlit 채팅 앱
  - RAG 모드 / Agent 모드 선택
  - 채팅 히스토리 표시
  - 파일 업로드 기능 (PDF)
- [ ] FastAPI 백엔드와 연동 (requests 또는 httpx)
- [ ] UI 스타일링 및 사용자 경험 개선

## 완료 조건
- `streamlit run deployment/streamlit_app.py`로 UI가 실행된다
- RAG/Agent 모드를 선택하여 대화할 수 있다
- 파일 업로드가 동작한다

## 참고
- `import streamlit as st`
- `st.chat_input()`, `st.chat_message()` 활용
- rag-project의 채팅 UI 경험 참고 (기능적으로 유사)
