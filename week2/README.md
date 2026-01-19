# Week 2: PyTorch 심화 + BERT (56시간)

> **목표:** CNN, LSTM, Transformers, BERT Fine-tuning 마스터

## 📅 주차 일정

### Day 8-10 (월-수): CNN + RNN/LSTM 마스터
**학습 시간:** 24시간
- Transfer Learning (ResNet)
- LSTM 시계열 예측
- Data Augmentation

### Day 11-14 (목-일): Transformers + BERT 실전
**학습 시간:** 32시간
- Hugging Face Transformers
- BERT Fine-tuning
- NLP 파이프라인

## 🎯 학습 목표

### 핵심 역량
- ✅ CNN Transfer Learning (ResNet)
- ✅ LSTM 시계열 예측
- ✅ BERT Fine-tuning
- ✅ Hugging Face 라이브러리

### 완성 프로젝트
4. **금융 문서 이미지 분류** - CNN
5. **주가 예측 LSTM** - 시계열
6. **금융 뉴스 감성 분석** - BERT
7. **금융 문서 Q&A** - Extractive QA

## 📚 학습 강의

### Day 8-10 (9시간 강의)
- Fast.ai Lesson 3-5 (CNN 중심)
- Stanford CS231n (핵심 강의만)
- PyTorch Lightning 튜토리얼

### Day 11-14 (12시간 강의)
- Hugging Face Course (Chapter 1-4)
- "Attention is All You Need" 논문
- BERT 구조 이해

## 🛠️ 필수 패키지 설치

```bash
conda activate ai-dev

# CNN & LSTM
conda install pytorch torchvision -c pytorch -y

# Transformers
pip install transformers datasets tokenizers
pip install accelerate
pip install sentencepiece

# 한국어 NLP
pip install konlpy
```

## 💻 실습 프로젝트

### Project 4: 금융 문서 이미지 분류 (Day 8-10)

**파일:** `document_classifier.py`

```python
"""
금융 문서 이미지 분류 (CNN Transfer Learning)
- 신분증, 계약서, 청구서 등 5개 클래스
- ResNet50 Transfer Learning
- Data Augmentation
- F1-Score 90% 이상 목표
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# 1. 데이터 전처리 & Augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. 모델 정의 (ResNet50 Transfer Learning)
class FinancialDocClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# 3. 학습 루프
def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%')
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_doc_classifier.pth')
        
        scheduler.step()
    
    return model

# 4. 실행
model = FinancialDocClassifier(num_classes=5)
# train_model(model, train_loader, val_loader)
```

**체크포인트:**
- [ ] Transfer Learning 구현
- [ ] Data Augmentation 적용
- [ ] 90% 이상 정확도
- [ ] 모델 저장/로드

---

### Project 5: 주가 예측 LSTM (Day 8-10)

**파일:** `stock_lstm.py`

```python
"""
주가 예측 LSTM
- 다변량 시계열 (시가, 고가, 저가, 거래량)
- 60일 데이터 → 다음날 예측
- RMSE, MAPE 평가
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 데이터 전처리
def prepare_data(df, seq_length=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length, 3])  # Close price
    
    return np.array(X), np.array(y), scaler

# 학습
def train_lstm(model, train_loader, num_epochs=50):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}')
    
    return model

# 평가 (RMSE, MAPE)
def evaluate(model, test_loader, scaler):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions.extend(output.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
    # Inverse transform
    predictions = scaler.inverse_transform(
        np.concatenate([np.zeros((len(predictions), 3)), 
                       np.array(predictions)], axis=1)
    )[:, 3]
    
    actuals = scaler.inverse_transform(
        np.concatenate([np.zeros((len(actuals), 3)), 
                       np.array(actuals).reshape(-1, 1)], axis=1)
    )[:, 3]
    
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print(f'RMSE: {rmse:.2f}, MAPE: {mape:.2f}%')
```

**체크포인트:**
- [ ] LSTM 구조 이해
- [ ] 시계열 데이터 전처리
- [ ] Walk-forward Validation
- [ ] RMSE < 5% 달성

---

### Project 6: 금융 뉴스 감성 분석 (Day 11-14)

**파일:** `financial_sentiment.py`

```python
"""
금융 뉴스 감성 분석 (BERT Fine-tuning)
- 모델: klue/bert-base
- 3-class: 긍정/중립/부정
- F1-Score 85% 이상
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset, Dataset
import torch

# 1. 데이터 준비
def prepare_dataset():
    # 금융 뉴스 데이터셋 (예시)
    data = {
        'text': [
            "주가가 급등하며 투자자들의 기대감이 높아지고 있다",
            "경제 위기로 인한 불확실성이 지속되고 있다",
            # ... more data
        ],
        'label': [2, 0, ...]  # 0: 부정, 1: 중립, 2: 긍정
    }
    
    dataset = Dataset.from_dict(data)
    return dataset.train_test_split(test_size=0.2)

# 2. 토크나이저 & 전처리
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# 3. 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3
)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 5. 평가 메트릭
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {"accuracy": acc, "f1": f1}

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# 7. 학습 & 평가
trainer.train()
results = trainer.evaluate()
print(results)

# 8. 모델 저장
model.save_pretrained("./financial_sentiment_model")
tokenizer.save_pretrained("./financial_sentiment_model")
```

**체크포인트:**
- [ ] BERT Fine-tuning 완료
- [ ] F1-Score 85% 이상
- [ ] 모델 저장/배포 준비
- [ ] Inference 파이프라인

---

### Project 7: 금융 문서 Q&A (Day 11-14)

**파일:** `document_qa.py`

```python
"""
금융 문서 Q&A (Extractive Question Answering)
- 계약서, 약관에서 정보 추출
- 모델: klue/roberta-large
"""

from transformers import pipeline

# 1. QA 파이프라인
qa_pipeline = pipeline(
    "question-answering",
    model="klue/roberta-large",
    tokenizer="klue/roberta-large"
)

# 2. 금융 계약서 예시
context = """
본 대출 계약의 이자율은 연 4.5%이며, 상환 기간은 36개월입니다. 
조기 상환 시 위약금은 없으나, 최소 6개월 이후부터 가능합니다.
대출 한도는 최대 5천만원이며, 담보는 부동산으로 설정됩니다.
"""

# 3. 질문 & 답변
questions = [
    "이자율은 얼마인가요?",
    "상환 기간은?",
    "조기 상환 위약금은?",
    "대출 한도는 얼마인가요?"
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}")

# 4. 배치 처리
def batch_qa(questions, context):
    results = []
    for q in questions:
        result = qa_pipeline(question=q, context=context)
        results.append({
            'question': q,
            'answer': result['answer'],
            'confidence': result['score']
        })
    return results
```

**체크포인트:**
- [ ] Extractive QA 이해
- [ ] 금융 문서 처리
- [ ] 배치 처리 구현
- [ ] 90% 이상 정확도

---

## ✅ Week 2 완료 체크리스트

### 프로젝트 완성도
- [ ] Project 4: 문서 분류 90%+ ✅
- [ ] Project 5: LSTM 예측 RMSE < 5% ✅
- [ ] Project 6: BERT 감성 분석 F1 85%+ ✅
- [ ] Project 7: 문서 Q&A 90%+ ✅

### 기술 습득
- [ ] Transfer Learning 마스터
- [ ] LSTM 시계열 예측
- [ ] BERT Fine-tuning
- [ ] Hugging Face 활용

### GitHub
- [ ] 4개 프로젝트 커밋
- [ ] 모델 파일 저장
- [ ] 학습 로그 정리

### 다음 주 준비
- [ ] Week 3 계획 확인
- [ ] LangChain 개념 예습
- [ ] OpenAI API 키 발급

## 📊 학습 시간 기록

| 일자 | 활동 | 시간 | 완료 |
|------|------|------|------|
| Day 8 | CNN 강의 + Transfer Learning | 8h | [ ] |
| Day 9 | 문서 분류 프로젝트 | 8h | [ ] |
| Day 10 | LSTM 주가 예측 | 8h | [ ] |
| Day 11 | Transformers 강의 | 8h | [ ] |
| Day 12 | BERT Fine-tuning | 8h | [ ] |
| Day 13 | 감성 분석 프로젝트 | 8h | [ ] |
| Day 14 | 문서 Q&A 완성 | 8h | [ ] |

---

**Week 2 완료 후 → Week 3 (BERT 심화)로 진행**
