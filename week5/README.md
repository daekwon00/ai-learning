# Week 5: 통합 프로젝트 + MLOps + 포트폴리오 (56시간)

> **목표:** AI 기반 금융 규제 준수 시스템 완성 + 포트폴리오 배포

## 📅 주차 일정

### Day 29-31 (월-수): 최종 프로젝트 통합
**학습 시간:** 24시간
- BERT + RAG 통합
- FastAPI 백엔드
- Streamlit 프론트엔드

### Day 32-33 (목-금): MLOps
**학습 시간:** 16시간
- MLflow 실험 추적
- Docker 컨테이너화
- docker-compose 배포

### Day 34-35 (토-일): 포트폴리오 완성
**학습 시간:** 16시간
- GitHub 저장소 정리
- README & 문서화
- 기술 블로그 작성

## 🎯 최종 목표

### 완성 프로젝트
**AI 기반 금융 규제 준수 시스템**
- 문서 분류 (BERT)
- RAG 검색 (LangChain)
- 규제 위반 분석 (GPT-4)
- 웹 인터페이스 (Streamlit)
- Docker 배포

### 포트폴리오
- GitHub 저장소 3개
- 기술 블로그 2편
- 프로젝트 데모 영상

## 🏗️ 최종 프로젝트: AI 기반 금융 규제 준수 시스템

### 시스템 아키텍처

```
┌──────────────────────────────────────────────┐
│         Frontend (Streamlit)                  │
│  - 문서 업로드                                 │
│  - Q&A 인터페이스                              │
│  - 대시보드                                    │
└───────────────┬──────────────────────────────┘
                │ HTTP/REST
                ▼
┌──────────────────────────────────────────────┐
│         FastAPI Backend                       │
│  - /upload    (문서 업로드)                    │
│  - /query     (RAG 검색)                       │
│  - /classify  (문서 분류)                      │
│  - /analyze   (규제 위반 분석)                 │
└──┬────────┬─────────┬──────────────────────┬─┘
   │        │         │                      │
   ▼        ▼         ▼                      ▼
┌────────┐ ┌───────┐ ┌──────────┐ ┌─────────────┐
│LangChain│ │PyTorch│ │ ChromaDB │ │   MLflow    │
│+ OpenAI │ │ BERT  │ │Vector DB │ │실험 추적     │
└────────┘ └───────┘ └──────────┘ └─────────────┘
```

### 프로젝트 구조

```
final_project/
├── README.md
├── docker-compose.yml
├── .env.example
│
├── backend/
│   ├── main.py                    # FastAPI 서버
│   ├── models/
│   │   ├── classifier_service.py  # BERT 분류
│   │   ├── rag_service.py         # RAG 검색
│   │   └── compliance_analyzer.py # 규제 분석
│   ├── utils/
│   │   └── document_processor.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app.py                     # Streamlit 앱
│   ├── requirements.txt
│   └── Dockerfile
│
├── mlops/
│   ├── train_classifier.py       # 모델 학습
│   ├── evaluate.py                # 모델 평가
│   └── mlflow_tracking.py         # 실험 추적
│
└── models/
    └── doc_classifier/            # 학습된 모델
```

## 💻 구현

### 1. 문서 분류 서비스 (backend/models/classifier_service.py)

```python
"""
BERT 기반 문서 분류 서비스
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DocumentClassifier:
    def __init__(self, model_path="./models/doc_classifier"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.categories = [
            "대출계약서", "보험약관", "투자설명서", 
            "금융규제문서", "기타"
        ]
    
    def classify(self, text: str) -> dict:
        """텍스트 분류"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        return {
            "category": self.categories[pred_class],
            "confidence": float(probs[0][pred_class].item()),
            "all_probabilities": {
                cat: float(prob) 
                for cat, prob in zip(self.categories, probs[0].tolist())
            }
        }
```

### 2. 규제 준수 분석기 (backend/models/compliance_analyzer.py)

```python
"""
GPT-4 기반 규제 위반 분석
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import os

class ComplianceAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.prompt = PromptTemplate(
            input_variables=["document_type", "content"],
            template="""
            당신은 금융 규제 전문가입니다.
            
            다음 {document_type} 문서를 분석하여 금융규제 위반 가능성을 검토하세요.
            
            문서 내용:
            {content}
            
            검토 항목:
            1. 금융소비자보호법 준수 여부
            2. 정보 공개 의무 충족 여부
            3. 불공정 조항 존재 여부
            
            JSON 형식으로 반환:
            {{
              "risk_level": "high/medium/low",
              "violations": ["위반 사항 1", "위반 사항 2", ...],
              "recommendations": ["권장 사항 1", "권장 사항 2", ...]
            }}
            
            JSON만 반환하세요.
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def analyze(self, document_type: str, content: str) -> dict:
        """규제 위반 분석"""
        try:
            # 토큰 제한을 위해 content 자르기
            content = content[:4000]
            
            result = self.chain.run(
                document_type=document_type,
                content=content
            )
            
            # JSON 파싱
            # GPT-4가 ```json ... ``` 형식으로 반환할 수 있음
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            
            return json.loads(result.strip())
        
        except Exception as e:
            return {
                "risk_level": "unknown",
                "violations": [f"분석 오류: {str(e)}"],
                "recommendations": []
            }
```

### 3. FastAPI 메인 서버 (backend/main.py)

```python
"""
FastAPI 통합 서버
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.classifier_service import DocumentClassifier
from models.rag_service import EnterpriseRAG
from models.compliance_analyzer import ComplianceAnalyzer
from utils.document_processor import extract_text
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Financial Compliance AI System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
classifier = DocumentClassifier()
rag = EnterpriseRAG()
analyzer = ComplianceAnalyzer()

# Request Models
class QueryRequest(BaseModel):
    question: str

# Endpoints

@app.get("/")
async def root():
    return {"message": "Financial Compliance AI API", "version": "1.0"}

@app.post("/api/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    """종합 문서 분석"""
    try:
        # 1. 파일 읽기
        content = await file.read()
        text = extract_text(content, file.filename)
        
        # 2. 문서 분류
        classification = classifier.classify(text)
        
        # 3. Vector DB 저장
        num_chunks = rag.add_documents([file.filename])
        
        # 4. 규제 위반 분석
        compliance = analyzer.analyze(
            classification['category'],
            text
        )
        
        return {
            "filename": file.filename,
            "classification": classification,
            "compliance_analysis": compliance,
            "chunks_added": num_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_rag(request: QueryRequest):
    """RAG 검색"""
    result = rag.query(request.question)
    return result

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4. Streamlit Frontend (frontend/app.py)

```python
"""
Streamlit 웹 인터페이스
"""

import streamlit as st
import requests
import os

st.set_page_config(
    page_title="AI 금융 규제 준수 시스템",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 AI 기반 금융 규제 준수 시스템")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ 설정")
    backend_url = st.text_input(
        "Backend URL",
        value="http://localhost:8000"
    )
    st.markdown("---")
    st.markdown("### 📊 시스템 정보")
    st.info("BERT 문서 분류 + GPT-4 규제 분석 + RAG 검색")

# Main Content
tab1, tab2 = st.tabs(["📄 문서 분석", "💬 Q&A"])

# Tab 1: 문서 분석
with tab1:
    st.header("문서 업로드 및 분석")
    
    uploaded_file = st.file_uploader(
        "문서를 업로드하세요 (PDF, DOCX)",
        type=['pdf', 'docx']
    )
    
    if uploaded_file:
        if st.button("분석 시작", type="primary"):
            with st.spinner("분석 중..."):
                try:
                    # API 호출
                    files = {'file': uploaded_file}
                    response = requests.post(
                        f"{backend_url}/api/analyze-document",
                        files=files
                    )
                    result = response.json()
                    
                    # 결과 표시
                    st.success("분석 완료!")
                    
                    # 문서 분류
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "문서 분류",
                            result['classification']['category']
                        )
                    with col2:
                        st.metric(
                            "분류 신뢰도",
                            f"{result['classification']['confidence']:.2%}"
                        )
                    
                    # 규제 위반 분석
                    st.subheader("⚠️ 규제 준수 분석")
                    compliance = result['compliance_analysis']
                    
                    # 위험도
                    risk_color = {
                        'high': '🔴',
                        'medium': '🟡',
                        'low': '🟢'
                    }
                    st.write(f"### 위험도: {risk_color.get(compliance['risk_level'], '⚪')} {compliance['risk_level'].upper()}")
                    
                    # 위반 사항
                    if compliance['violations']:
                        st.write("**위반 가능성:**")
                        for v in compliance['violations']:
                            st.warning(v)
                    
                    # 권장사항
                    if compliance['recommendations']:
                        st.write("**개선 권장사항:**")
                        for r in compliance['recommendations']:
                            st.info(r)
                
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

# Tab 2: Q&A
with tab2:
    st.header("문서 질의응답 (RAG)")
    
    question = st.text_input("질문을 입력하세요")
    
    if st.button("질문하기"):
        if question:
            with st.spinner("답변 생성 중..."):
                try:
                    response = requests.post(
                        f"{backend_url}/api/query",
                        json={"question": question}
                    )
                    result = response.json()
                    
                    # 답변
                    st.write("### 💡 답변:")
                    st.write(result['answer'])
                    
                    # 출처
                    with st.expander("📚 출처 보기"):
                        for i, source in enumerate(result['sources'], 1):
                            st.write(f"**Source {i}:**")
                            st.write(source['content'])
                            st.write(f"*Metadata:* {source['metadata']}")
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"오류: {str(e)}")
        else:
            st.warning("질문을 입력해주세요")

# Footer
st.markdown("---")
st.markdown("**개발:** YDK | **기술 스택:** BERT, GPT-4, LangChain, ChromaDB")
```

### 5. Docker Compose (docker-compose.yml)

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./models:/app/models
      - ./chroma_db:/app/chroma_db
    command: uvicorn main:app --host 0.0.0.0 --port 8000
  
  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
    environment:
      - BACKEND_URL=http://backend:8000
    command: streamlit run app.py
  
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow
    command: mlflow server --host 0.0.0.0 --port 5000
```

### 6. MLflow 실험 추적 (mlops/train_classifier.py)

```python
"""
MLflow를 사용한 모델 학습 추적
"""

import mlflow
import mlflow.pytorch
from transformers import Trainer, TrainingArguments
import torch

# MLflow 실험 설정
mlflow.set_experiment("financial_doc_classifier")

with mlflow.start_run(run_name="bert_finetuning_v1"):
    # 하이퍼파라미터
    params = {
        "model": "klue/bert-base",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "max_length": 512
    }
    mlflow.log_params(params)
    
    # 학습
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=params['epochs'],
        per_device_train_batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 학습 실행
    trainer.train()
    
    # 평가
    results = trainer.evaluate()
    
    # 메트릭 로깅
    mlflow.log_metrics(results)
    
    # 모델 저장
    mlflow.pytorch.log_model(model, "model")
    
    print(f"Model saved with run_id: {mlflow.active_run().info.run_id}")
```

## ✅ Week 5 완료 체크리스트

### Day 29-31 (통합 프로젝트)
- [ ] BERT + RAG 통합
- [ ] FastAPI 백엔드 완성
- [ ] Streamlit 프론트엔드 완성
- [ ] 종단간 테스트

### Day 32-33 (MLOps)
- [ ] MLflow 실험 추적
- [ ] Docker 이미지 빌드
- [ ] docker-compose 배포
- [ ] 성능 모니터링

### Day 34-35 (포트폴리오)
- [ ] GitHub 3개 저장소 정리
  - financial-ai-compliance-system
  - pytorch-financial-models
  - langchain-rag-examples
- [ ] README 작성 (프로 수준)
- [ ] 기술 블로그 2편
  - "BERT Fine-tuning 실전"
  - "엔터프라이즈 RAG 구축기"
- [ ] 데모 영상 제작

## 📊 최종 점검

### 기술 스택 마스터
- [ ] Python ✅
- [ ] PyTorch ✅
- [ ] BERT ✅
- [ ] LangChain ✅
- [ ] FastAPI ✅
- [ ] Docker ✅

### 프로젝트 포트폴리오
- [ ] 10개 실습 프로젝트
- [ ] 1개 통합 프로젝트
- [ ] GitHub 저장소
- [ ] 기술 블로그

## 🎓 5주 학습 완료!

**축하합니다! 🎉**

280시간의 집중 학습을 완료하셨습니다!

### 달성한 역량
✅ PyTorch 딥러닝 개발
✅ BERT Fine-tuning
✅ LangChain RAG 시스템
✅ 엔터프라이즈 AI 시스템 구축
✅ MLOps 기초

### 다음 단계
1. 면접 준비
2. 채용 공고 지원
3. 포트폴리오 공유

---

**당신은 이제 Enterprise AI System Developer입니다! 💪**
