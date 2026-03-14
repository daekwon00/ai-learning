# Phase 3 - Step 04: 결과 시각화 + 정리

## 목표
- Fine-tuning 결과를 시각화하고 Phase 3를 정리한다

## 작업 항목
- [ ] Confusion Matrix 시각화
- [ ] 클래스별 Precision/Recall/F1 리포트
- [ ] 학습 곡선 (loss, accuracy by epoch)
- [ ] 예측 샘플 분석 (맞은 것/틀린 것)
- [ ] `pytorch-bert/bert_finetuning.ipynb` — 전체 파이프라인 통합 노트북
- [ ] `pytorch-bert/summary.md` — Phase 3 학습 요약
- [ ] `pytorch-bert/README.md` 업데이트

## 완료 조건
- Confusion Matrix와 학습 곡선이 시각화된다
- 노트북이 처음부터 끝까지 실행된다
- README.md에 결과가 기록되어 있다

## 참고
- `from sklearn.metrics import confusion_matrix, classification_report`
- Phase 0의 시각화 경험 활용 (matplotlib, seaborn)
