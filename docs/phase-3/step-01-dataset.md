# Phase 3 - Step 01: 데이터셋 준비 + 탐색적 분석

## 목표
- 금융 뉴스 감성 분석용 데이터셋을 준비한다
- 데이터 분포와 특성을 파악한다

## 작업 항목
- [ ] `pytorch-bert/requirements.txt` 작성 및 패키지 설치
- [ ] 데이터셋 선택 및 다운로드
  - 옵션 A: Financial PhraseBank (영어, HuggingFace datasets)
  - 옵션 B: 한국어 금융 뉴스 데이터셋
- [ ] 탐색적 분석 (EDA)
  - 클래스 분포 (positive/negative/neutral)
  - 텍스트 길이 분포
  - 워드클라우드 또는 빈출 단어
- [ ] Train/Validation/Test 분할

## 완료 조건
- 데이터셋이 로드되고 3-class 분포가 확인된다
- Train/Val/Test 분할이 완료된다

## 참고
- `from datasets import load_dataset`
- Financial PhraseBank: `load_dataset("financial_phrasebank", "sentences_allagree")`
- Phase 0의 EDA 경험 활용 (credit_scoring.ipynb)
