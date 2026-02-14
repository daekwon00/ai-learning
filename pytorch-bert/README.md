# Phase 3: PyTorch + HuggingFace BERT Fine-tuning

> BERT 모델을 금융 뉴스 감성 분석에 Fine-tuning

## 학습 목표

- **필수**: BERT Fine-tuning 금융 뉴스 감성 분석 (3-class: positive/negative/neutral)
  - HuggingFace Trainer API
  - Financial PhraseBank 또는 한국어 금융 뉴스 데이터셋
  - 평가: Accuracy, F1, Confusion Matrix
- **선택**: CNN 심화 / LSTM 주가 예측

## 디렉토리 구조

```
pytorch-bert/
├── README.md
├── requirements.txt
├── bert_finetuning.ipynb   # 메인 노트북
├── data/                   # 학습 데이터
├── models/                 # 저장된 모델
└── results/                # 평가 결과
```

## 환경

```bash
conda activate ai-dev
pip install transformers datasets accelerate
```

## 상태: 미착수
