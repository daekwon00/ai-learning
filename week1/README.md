# Week 1: Python + ML 기초 (56시간)

> **목표:** Python 데이터 과학 + Scikit-learn + PyTorch 기초 완성

## 📅 주차 일정

### Day 1-2 (월-화): Python 데이터 과학 도구 마스터
**학습 시간:** 16시간
- NumPy 고급 연산
- Pandas DataFrame 조작
- Matplotlib/Seaborn 시각화

### Day 3-4 (수-목): Scikit-learn + 전통 ML
**학습 시간:** 16시간
- 데이터 전처리
- Logistic Regression, Random Forest
- 모델 평가 (Precision, Recall, F1)

### Day 5-7 (금-일): PyTorch 기초 + 딥러닝 입문
**학습 시간:** 24시간
- Tensor 연산
- 신경망 구현
- MNIST 분류

## 🎯 학습 목표

### 핵심 역량
- ✅ Pandas로 금융 데이터 전처리
- ✅ NumPy 벡터 연산 마스터
- ✅ Scikit-learn ML 파이프라인
- ✅ PyTorch 신경망 구현

### 완성 프로젝트
1. **주식 데이터 분석** - Pandas, NumPy
2. **신용 평가 모델** - Scikit-learn
3. **PyTorch MNIST** - PyTorch 기초

## 📚 학습 강의

### Day 1-2 (6시간 강의)
- Coursera: "Python for Data Science" (속성)
- YouTube: 생활코딩 Numpy/Pandas (핵심만)

### Day 3-4 (6시간 강의)
- Coursera: Machine Learning (Andrew Ng) Week 1-2
- Fast.ai: Tabular Learner

### Day 5-7 (9시간 강의)
- PyTorch 공식 튜토리얼 (60분 블리츠)
- DeepLearning.AI: Neural Networks Basics
- Fast.ai Lesson 1-2

## 🛠️ 필수 패키지 설치

```bash
conda activate ai-dev

# Day 1-2
pip install numpy pandas matplotlib seaborn scikit-learn jupyter yfinance

# Day 3-4
conda install scikit-learn -y

# Day 5-7
conda install pytorch torchvision torchaudio -c pytorch -y
```

## 💻 실습 프로젝트

### Project 1: 금융 데이터 분석 (Day 1-2)

**파일:** `stock_analysis.ipynb`

```python
"""
주식 데이터 분석 프로젝트
1. yfinance로 은행 주가 수집
2. Pandas 전처리
3. NumPy 지표 계산 (이동평균, 볼린저밴드)
4. Matplotlib 시각화
5. 상관관계 분석
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 신한은행, 국민은행 주가
tickers = ['055550.KS', '105560.KS']
data = yf.download(tickers, start='2023-01-01', end='2024-01-01')

# 이동평균선
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA60'] = data['Close'].rolling(window=60).mean()

# 볼린저 밴드
data['Upper'] = data['MA20'] + 2*data['Close'].rolling(window=20).std()
data['Lower'] = data['MA20'] - 2*data['Close'].rolling(window=20).std()

# 시각화
plt.figure(figsize=(15, 7))
plt.plot(data.index, data['Close'], label='Close Price')
plt.plot(data.index, data['MA20'], label='MA20')
plt.plot(data.index, data['MA60'], label='MA60')
plt.legend()
plt.show()
```

**체크포인트:**
- [ ] yfinance로 데이터 수집
- [ ] Pandas DataFrame 조작 숙련
- [ ] NumPy 계산 구현
- [ ] 시각화 완성

---

### Project 2: 신용 평가 모델 (Day 3-4)

**파일:** `credit_scoring.ipynb`

```python
"""
신용 평가 모델 프로젝트
Dataset: Kaggle - Credit Card Default
1. 데이터 전처리 (결측치, 스케일링)
2. Train/Test 분할
3. Random Forest 학습
4. 모델 평가
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 데이터 로드
df = pd.read_csv('credit_data.csv')

# 전처리
X = df.drop('default', axis=1)
y = df['default']

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 평가
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5)
print(f'CV Scores: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
```

**체크포인트:**
- [ ] 데이터 전처리 완료
- [ ] ML 파이프라인 구축
- [ ] 모델 평가 지표 이해
- [ ] Cross-validation 적용

---

### Project 3: PyTorch MNIST (Day 5-7)

**파일:** `pytorch_basics.ipynb`

```python
"""
PyTorch MNIST 분류
1. Tensor 연산 마스터
2. 신경망 구현
3. 학습/검증 파이프라인
4. 95% 이상 정확도 달성
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 데이터 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    './data', train=False, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 신경망 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 3. 학습 루프
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

# 4. 평가
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 5. 실행
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
```

**체크포인트:**
- [ ] PyTorch Tensor 연산 이해
- [ ] 신경망 구조 구현
- [ ] 학습/검증 파이프라인
- [ ] 95% 이상 정확도 달성

---

## ✅ Week 1 완료 체크리스트

### 프로젝트 완성도
- [ ] Project 1: 주식 데이터 분석 ✅
- [ ] Project 2: 신용 평가 모델 ✅
- [ ] Project 3: PyTorch MNIST 95%+ ✅

### 기술 습득
- [ ] Pandas DataFrame 자유자재
- [ ] NumPy 벡터 연산 마스터
- [ ] Scikit-learn 파이프라인
- [ ] PyTorch Tensor & 신경망

### GitHub
- [ ] 3개 프로젝트 커밋
- [ ] README 작성
- [ ] 학습 노트 정리

### 다음 주 준비
- [ ] Week 2 계획 확인
- [ ] CNN 개념 예습
- [ ] LSTM 이론 학습

## 🔗 참고 자료

- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Scikit-learn 튜토리얼](https://scikit-learn.org/stable/tutorial/)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [Kaggle Credit Card Dataset](https://www.kaggle.com/datasets)

## 📊 학습 시간 기록

| 일자 | 활동 | 시간 | 완료 |
|------|------|------|------|
| Day 1 | NumPy, Pandas 강의 + 실습 | 8h | [ ] |
| Day 2 | 주식 분석 프로젝트 | 8h | [ ] |
| Day 3 | Scikit-learn 강의 | 8h | [ ] |
| Day 4 | 신용 평가 모델 | 8h | [ ] |
| Day 5 | PyTorch 기초 | 8h | [ ] |
| Day 6 | MNIST 프로젝트 | 8h | [ ] |
| Day 7 | 복습 & 정리 | 8h | [ ] |

---

**Week 1 완료 후 → Week 2 (PyTorch 심화 + BERT)로 진행**
