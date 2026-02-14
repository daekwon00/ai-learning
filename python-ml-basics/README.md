# Week 1: Python + ML 기초

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

> 포트폴리오용 Week 1 결과물: 데이터 분석, 전통 ML, PyTorch 기초를 단계적으로 정리

---

## 1) 학습 목표 및 달성 내용

### 학습 목표

- 금융 도메인 데이터를 활용해 데이터 분석/모델링 기초를 다진다.
- Scikit-learn 파이프라인과 평가 지표를 이해한다.
- PyTorch 텐서/신경망/MNIST 분류 흐름을 완성한다.

### 달성 내용

- 주식 데이터 분석 파이프라인 구축 (수집 → 전처리 → 지표 → 시각화)
- 신용평가 ML 파이프라인 구성 (EDA → 전처리 → 모델 비교 → 최적화)
- PyTorch 기초 학습 + MNIST CNN 학습 및 상세 평가

---

## 2) 프로젝트 목록

- `stock_analysis.ipynb`: 주식 데이터 분석 (Pandas, Matplotlib)
- `credit_scoring.ipynb`: 신용평가 ML (Scikit-learn)
- `pytorch_basics.ipynb`: PyTorch 기초 + MNIST

---

## 3) 프로젝트 설명 및 실행 방법

### A. `stock_analysis.ipynb` — 주식 데이터 분석

- 사용 기술: `yfinance`, `pandas`, `matplotlib`, `seaborn`
- 주요 학습 내용
  - 한국 주식 3종목 데이터 수집 (삼성전자/하이닉스/NAVER)
  - 이동평균/변동성/기술지표 산출 및 시각화
  - 리스크/수익률 분석과 종목 비교
- 실행 방법
  - Jupyter에서 `stock_analysis.ipynb` 실행

### B. `credit_scoring.ipynb` — 신용평가 ML

- 사용 기술: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- 주요 학습 내용
  - EDA 및 전처리 파이프라인 구축
  - 여러 모델 비교 및 최적화(GridSearch)
  - 모델 해석(Feature Importance/SHAP) 및 저장
- 실행 방법
  - Kaggle 데이터가 있을 경우 `week1/data/cs-training.csv`에 저장 후 실행
  - 없으면 샘플 데이터로 자동 생성

### C. `pytorch_basics.ipynb` — PyTorch 기초 + MNIST

- 사용 기술: `torch`, `torchvision`, `matplotlib`
- 주요 학습 내용
  - 텐서 연산/자동미분/선형회귀
  - nn.Module 기반 신경망과 옵티마이저 비교
  - MNIST 데이터 로딩, CNN 학습, 상세 평가
- 실행 방법
  - `pytorch_basics.ipynb` 실행 (MNIST 자동 다운로드)

---

## 4) 설치 방법

```bash
conda activate ai-dev
pip install -r week1/requirements.txt
```

---

## 5) 디렉토리 구조

```
week1/
├── README.md
├── summary.md
├── requirements.txt
├── stock_analysis.ipynb
├── credit_scoring.ipynb
├── pytorch_basics.ipynb
├── data/
└── outputs/
```

---

## 6) 스크린샷 (차트 예시)

> 실제 실행 후 생성된 이미지 경로를 연결하세요.

![주식 종가 추이](outputs/close_prices_3stocks.png)
![신용평가 EDA 예시](outputs/credit_eda_example.png)
![MNIST 예측 예시](outputs/mnist_confusion_matrix.png)

---

## 7) 학습 후기 및 다음 단계

### 학습 후기

- 데이터 전처리와 시각화가 모델 성능과 해석의 핵심임을 체감
- 전통 ML과 딥러닝의 학습 루프 차이를 명확히 이해
- 금융 도메인에서는 **설명력 + 성능**의 균형이 중요함을 확인

### 다음 단계

- PyTorch 심화(CNN 성능 개선, 정규화/스케줄러 적용)
- 불균형 데이터 처리(SMOTE, class weight)
- 모델 해석/검증 체계 고도화(캘리브레이션, 오류 분석)

---

## 8) 요약 노트

- 1주차 요약: `summary.md`
