# Phase 3 - Step 02: BERT 토크나이징 + 데이터 전처리

## 목표
- BERT 토크나이저로 텍스트를 토큰화한다
- PyTorch Dataset으로 변환한다

## 작업 항목
- [ ] BERT 모델 선택 (`bert-base-uncased` 또는 한국어 BERT)
- [ ] 토크나이저 로드 및 텍스트 토큰화
  - `AutoTokenizer.from_pretrained()`
  - max_length, padding, truncation 설정
- [ ] PyTorch Dataset 클래스 구현 또는 HuggingFace datasets 형식 활용
- [ ] DataLoader 구성 (batch_size, shuffle)

## 완료 조건
- 토큰화된 데이터가 모델 입력 형식 (input_ids, attention_mask, labels)을 갖는다
- DataLoader에서 배치를 정상적으로 가져올 수 있다

## 참고
- `from transformers import AutoTokenizer`
- M4 Mac: `device="mps"` 사용 가능
