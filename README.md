# AI Python 개발자 학습 프로젝트

> 5주 집중 부트캠프 (하루 8시간 × 5주 = 280시간)

## 📋 목표
- Python 기반 AI/ML 개발 역량 구축 (PyTorch, BERT)
- LLM 플랫폼 실무 경험 (OpenAI API, LangChain)
- RAG 시스템 구축 및 AI Agent 개발
- MLOps 기초 및 포트폴리오 완성

## ⏰ 학습 강도
- **기간:** 5주 (2026.01.20 ~ 2026.02.24)
- **일일 투자:** 8시간
- **총 학습량:** 280시간

## 📅 일일 학습 스케줄

```plaintext
평일 (월-금):
09:00-12:00 (3시간) - 이론 학습 + 온라인 강의
12:00-13:00 (1시간) - 점심 + 기술 블로그
13:00-16:00 (3시간) - 실습 코딩
16:00-16:30 (30분)  - 휴식
16:30-18:30 (2시간) - 프로젝트 개발
18:30-19:00 (30분)  - 정리 & GitHub 커밋

주말 (토-일):
10:00-13:00 (3시간) - 주간 프로젝트 개발
13:00-14:00 (1시간) - 점심
14:00-17:00 (3시간) - 프로젝트 완성
17:00-18:00 (1시간) - 코드 리뷰 & 문서화
```

## 🛠️ 환경 설정

### Conda 환경 복원
```bash
conda env create -f ai-dev-environment.yml
conda activate ai-dev
```

### 새로 환경 만들 경우
```bash
conda create -n ai-dev python=3.11 -y
conda activate ai-dev

# 기본 패키지
conda install numpy pandas scikit-learn jupyter matplotlib seaborn -y

# 딥러닝 프레임워크
conda install pytorch torchvision torchaudio -c pytorch -y

# NLP & LLM
pip install transformers datasets tokenizers accelerate
pip install langchain langchain-openai langchain-community openai
pip install chromadb sentence-transformers faiss-cpu

# 문서 처리
pip install pypdf python-docx

# FastAPI & 서빙
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv

# MLOps
pip install mlflow streamlit

# 유틸리티
pip install yfinance plotly tqdm black
```

## 📚 주차별 학습 내용

### Week 1: Python + ML 기초 (56시간)
**Day 1-2 (월-화):** NumPy, Pandas, 금융 데이터 분석
**Day 3-4 (수-목):** Scikit-learn, 신용평가 모델
**Day 5-7 (금-일):** PyTorch 기초, MNIST

**프로젝트:**
1. 주식 데이터 분석
2. 신용 평가 모델
3. PyTorch MNIST

### Week 2: PyTorch 심화 + BERT (56시간)
**Day 8-10 (월-수):** CNN, LSTM
**Day 11-14 (목-일):** Transformers, BERT Fine-tuning

**프로젝트:**
4. 금융 문서 이미지 분류 (CNN)
5. 주가 예측 LSTM
6. 금융 뉴스 감성 분석 (BERT)
7. 금융 문서 Q&A

### Week 3: BERT + NLP 심화 (56시간)
**Day 15-17 (월-수):** BERT 고급, NLP 파이프라인
**Day 18-21 (목-일):** 실전 금융 NLP 프로젝트

**프로젝트:**
- BERT 기반 문서 분류
- Question Answering 시스템
- 감성 분석 고도화

### Week 4: LangChain + RAG (56시간)
**Day 22-24 (월-수):** LangChain, OpenAI API
**Day 25-28 (목-일):** RAG 시스템 구축

**프로젝트:**
8. LangChain 챗봇
9. LangChain Agent
10. 엔터프라이즈 RAG 시스템

### Week 5: 통합 프로젝트 + MLOps (56시간)
**Day 29-31 (월-수):** 최종 프로젝트 통합
**Day 32-33 (목-금):** MLOps (Docker, MLflow)
**Day 34-35 (토-일):** 포트폴리오 완성

**프로젝트:**
- AI 기반 금융 규제 준수 시스템
- Docker 배포
- GitHub 포트폴리오

## 🗂️ 디렉토리 구조

```
ai-learning/
├── README.md                   # 프로젝트 설명
├── .gitignore                  # Git 제외 파일
├── ai-dev-environment.yml      # Conda 환경 설정
├── requirements.txt            # pip 패키지 목록
│
├── week1/                      # Python + ML 기초
│   ├── README.md
│   ├── stock_analysis.ipynb
│   ├── credit_scoring.ipynb
│   └── pytorch_basics.ipynb
│
├── week2/                      # PyTorch 심화 + BERT
│   ├── README.md
│   ├── document_classifier.py
│   ├── stock_lstm.py
│   ├── financial_sentiment.py
│   └── document_qa.py
│
├── week3/                      # BERT + NLP 심화
│   ├── README.md
│   └── (Week 2 내용 심화)
│
├── week4/                      # LangChain + RAG
│   ├── README.md
│   ├── financial_chatbot.py
│   ├── langchain_agent.py
│   └── enterprise_rag/
│       ├── rag_engine.py
│       ├── main.py (FastAPI)
│       └── requirements.txt
│
├── week5/                      # 통합 프로젝트 + MLOps
│   ├── README.md
│   └── final_project/
│       ├── backend/
│       ├── frontend/
│       ├── mlops/
│       ├── docker-compose.yml
│       └── README.md
│
├── notebooks/                  # Jupyter 실습 노트북
├── datasets/                   # 학습용 데이터셋
├── models/                     # 저장된 모델 파일
└── scripts/                    # 유틸리티 스크립트
```

## 🎯 최종 포트폴리오 프로젝트

### 프로젝트명: AI 기반 금융 규제 준수 시스템

**기술 스택:**
- Backend: Python, FastAPI
- ML/AI: PyTorch (BERT), LangChain, OpenAI GPT-4
- Vector DB: ChromaDB
- Frontend: Streamlit
- MLOps: MLflow, Docker

**주요 기능:**
1. 문서 분류 (BERT Fine-tuned)
2. RAG 기반 의미 검색
3. 규제 위반 분석 (GPT-4)
4. 대화 히스토리 관리
5. 답변 출처 추적 (Citation)

**GitHub 저장소:**
- `financial-ai-compliance-system` (메인 프로젝트)
- `pytorch-financial-models` (모델 모음)
- `langchain-rag-examples` (RAG 예제)

## ✅ 주간 체크리스트

**매주 금요일 점검:**
- [ ] 해당 주차 강의 완료 (20-25시간)
- [ ] 실습 코드 GitHub 커밋
- [ ] 주간 프로젝트 완성
- [ ] 학습 내용 정리 (블로그/노션)
- [ ] 다음 주 학습 계획 수립

## 🎓 학습 리소스

### 온라인 강의
- **Coursera:** Machine Learning (Andrew Ng)
- **Fast.ai:** Practical Deep Learning
- **DeepLearning.AI:** LangChain Series
- **HuggingFace:** NLP Course

### 추천 도서 (점심시간)
- PyTorch 공식 문서
- LangChain 공식 문서
- Hugging Face Blog
- Arxiv 논문 (NLP/LLM)

## 📊 학습 진행 상황

### Week 1
- [ ] Day 1-2: NumPy, Pandas
- [ ] Day 3-4: Scikit-learn
- [ ] Day 5-7: PyTorch 기초
- [ ] 프로젝트 3개 완성

### Week 2
- [ ] Day 8-10: CNN, LSTM
- [ ] Day 11-14: BERT
- [ ] 프로젝트 4개 완성

### Week 3
- [ ] Day 15-17: BERT 심화
- [ ] Day 18-21: NLP 프로젝트
- [ ] BERT 마스터

### Week 4
- [ ] Day 22-24: LangChain
- [ ] Day 25-28: RAG
- [ ] RAG 시스템 완성

### Week 5
- [ ] Day 29-31: 통합 프로젝트
- [ ] Day 32-33: MLOps
- [ ] Day 34-35: 포트폴리오
- [ ] 최종 배포 완료

## 🚀 시작하기

```bash
# 1. 저장소 클론
cd ~/ai-learning

# 2. 환경 활성화
conda activate ai-dev

# 3. Week 1 시작
cd week1
jupyter lab
```

## 📧 문의

**작성자:** YDK  
**시작일:** 2026.01.20  
**목표:** Enterprise AI System Developer

---

**화이팅! 5주 후 당신은 AI 개발자입니다! 🔥**
