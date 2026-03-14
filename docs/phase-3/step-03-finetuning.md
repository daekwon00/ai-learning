# Phase 3 - Step 03: Fine-tuning 학습 + 평가

## 목표
- BERT 모델을 금융 뉴스 감성 분석에 Fine-tuning한다
- 학습 결과를 평가한다

## 작업 항목
- [ ] BERT for Sequence Classification 모델 로드
  - `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)`
- [ ] HuggingFace Trainer 설정
  - TrainingArguments (learning_rate, epochs, batch_size, eval_strategy)
  - compute_metrics 함수 (accuracy, f1)
- [ ] 학습 실행 및 모니터링
- [ ] 평가 메트릭 확인: Accuracy, F1 (macro/weighted)
- [ ] 모델 저장 (`pytorch-bert/models/`)

## 완료 조건
- Fine-tuning이 완료되고 validation accuracy가 합리적인 수준이다
- 모델이 저장된다

## 참고
- `from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments`
- M4 Mac MPS 가속 활용
- Phase 0의 PyTorch 학습 루프 경험 활용
