# Week 4: LangChain + RAG 시스템 (56시간)

> **목표:** LangChain 마스터 + 엔터프라이즈 RAG 시스템 구축

## 📅 주차 일정

### Day 22-24 (월-수): LangChain + OpenAI API
**학습 시간:** 24시간
- LangChain 기초
- Memory & Chain
- Agent & Tools

### Day 25-28 (목-일): RAG 시스템 구축
**학습 시간:** 32시간
- Vector Database
- Document Loading
- RAG 파이프라인
- FastAPI 서버

## 🎯 학습 목표

### 핵심 역량
- ✅ LangChain 완벽 숙련
- ✅ OpenAI API 활용
- ✅ Vector Database (ChromaDB)
- ✅ RAG 시스템 구축
- ✅ FastAPI 서버 구축

### 완성 프로젝트
8. **LangChain 금융 챗봇**
9. **LangChain Agent (도구 활용)**
10. **엔터프라이즈 RAG 시스템** (핵심 포트폴리오)

## 📚 학습 강의

### Day 22-24 (9시간)
- DeepLearning.AI: ChatGPT Prompt Engineering
- DeepLearning.AI: LangChain for LLM Development
- DeepLearning.AI: Building Systems with ChatGPT API

### Day 25-28 (12시간)
- DeepLearning.AI: Building RAG Applications
- Vector Databases 완벽 가이드
- ChromaDB 튜토리얼

## 🛠️ 필수 패키지 설치

```bash
conda activate ai-dev

# LangChain
pip install langchain langchain-openai langchain-community
pip install openai

# Vector Database
pip install chromadb sentence-transformers faiss-cpu

# Document Loaders
pip install pypdf python-docx

# FastAPI
pip install fastapi uvicorn pydantic python-dotenv

# 유틸리티
pip install tiktoken
```

## 🔑 OpenAI API 설정

```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

```python
# .env 로드
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

## 💻 실습 프로젝트

### Project 8: LangChain 금융 챗봇 (Day 22-24)

**파일:** `financial_chatbot.py`

```python
"""
금융 상담 챗봇
- Memory 관리 (대화 히스토리)
- Chain 구성 (프롬프트 체인)
- 금융 지식 기반 응답
"""

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 1. LLM 초기화
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-4",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 2. 프롬프트 템플릿
system_template = """당신은 27년 경력의 전문 금융 상담사입니다.

전문성:
- 은행, 보험, 투자 전 분야 경험
- 금융소비자보호법 준수
- 리스크 관리 전문가

규칙:
✅ 정확한 금융 정보만 제공
✅ 불확실한 경우 "확인이 필요합니다" 명시
✅ 법적 조언은 회피 (전문가 상담 권장)
✅ 친절하고 전문적인 어조

대화 히스토리:
{history}
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# 3. Memory 설정
memory = ConversationBufferMemory(return_messages=True)

# 4. Conversation Chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# 5. 대화 실행
def chat(user_input):
    response = conversation.predict(input=user_input)
    return response

# 사용 예시
if __name__ == "__main__":
    print(chat("안녕하세요, 주택담보대출에 대해 궁금합니다."))
    print(chat("금리가 어떻게 되나요?"))
    print(chat("필요한 서류는 무엇인가요?"))
```

**체크포인트:**
- [ ] LangChain Chain 이해
- [ ] Memory 관리 구현
- [ ] 프롬프트 엔지니어링
- [ ] 대화 품질 테스트

---

### Project 9: LangChain Agent (Day 22-24)

**파일:** `langchain_agent.py`

```python
"""
LangChain Agent - 도구 활용
- 웹 검색 (주가, 뉴스)
- 계산기
- 데이터베이스 조회
"""

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
import yfinance as yf

# 1. LLM
llm = ChatOpenAI(temperature=0, model="gpt-4")

# 2. 도구 정의

# 도구 1: 주가 조회
def get_stock_price(ticker):
    """주가 조회 도구"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('currentPrice', 'N/A')
        return f"{ticker} 현재가: {current_price}원"
    except:
        return "주가 조회 실패"

# 도구 2: 계산기
def calculator(expression):
    """계산기 도구"""
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except:
        return "계산 오류"

# 도구 3: 웹 검색 (SerpAPI 필요)
# search = SerpAPIWrapper()

# 3. Tools 리스트
tools = [
    Tool(
        name="StockPrice",
        func=get_stock_price,
        description="주식 티커로 현재 주가를 조회합니다. 입력: 티커 심볼 (예: 005930.KS)"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="수학 계산을 수행합니다. 입력: 수식 (예: 100 * 1.05)"
    ),
    # Tool(
    #     name="Search",
    #     func=search.run,
    #     description="최신 뉴스와 정보를 검색합니다."
    # ),
]

# 4. Agent 초기화
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# 5. Agent 실행
def run_agent(query):
    try:
        result = agent.run(query)
        return result
    except Exception as e:
        return f"오류: {str(e)}"

# 사용 예시
if __name__ == "__main__":
    # 질문 1: 주가 조회
    print(run_agent("삼성전자(005930.KS) 주가는 얼마인가요?"))
    
    # 질문 2: 계산
    print(run_agent("1억원에 연 4.5% 이자를 36개월간 받으면 총 얼마인가요?"))
    
    # 질문 3: 복합 질문
    print(run_agent("삼성전자 주가가 작년 대비 몇 % 상승했나요?"))
```

**체크포인트:**
- [ ] Agent 개념 이해
- [ ] Tool 작성 및 등록
- [ ] 복잡한 질문 처리
- [ ] 에러 핸들링

---

### Project 10: 엔터프라이즈 RAG 시스템 (Day 25-28) ⭐

**프로젝트 구조:**
```
enterprise_rag/
├── rag_engine.py          # RAG 핵심 로직
├── main.py                # FastAPI 서버
├── document_loader.py     # 문서 로딩
├── vector_store.py        # Vector DB 관리
├── requirements.txt
└── .env
```

#### 1. RAG Engine (`rag_engine.py`)

```python
"""
엔터프라이즈 RAG 시스템 핵심
- PDF/Word 문서 임베딩
- 의미 검색 (Semantic Search)
- 답변 출처 추적
"""

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from typing import List
import os

class EnterpriseRAG:
    def __init__(self, persist_directory="./chroma_db"):
        """RAG 시스템 초기화"""
        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Vector Store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )
    
    def add_documents(self, file_paths: List[str]) -> int:
        """문서 추가 및 임베딩"""
        documents = []
        
        for file_path in file_paths:
            print(f"Loading: {file_path}")
            
            # 파일 타입별 로더
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue
            
            docs = loader.load()
            documents.extend(docs)
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        
        # Vector DB에 저장
        self.vectorstore.add_documents(splits)
        self.vectorstore.persist()
        
        print(f"Added {len(splits)} chunks to vector store")
        return len(splits)
    
    def query(self, question: str) -> dict:
        """질의 및 답변"""
        result = self.qa_chain({"query": question})
        
        # 출처 정보 추출
        sources = []
        for doc in result['source_documents']:
            sources.append({
                'content': doc.page_content[:200] + "...",
                'metadata': doc.metadata
            })
        
        return {
            'answer': result['result'],
            'sources': sources
        }
    
    def clear_database(self):
        """Vector DB 초기화"""
        # ChromaDB 클라이언트를 통해 컬렉션 삭제
        try:
            self.vectorstore._client.delete_collection(
                self.vectorstore._collection.name
            )
            print("Database cleared")
        except:
            print("Failed to clear database")
```

#### 2. FastAPI 서버 (`main.py`)

```python
"""
FastAPI 서버
- 문서 업로드 API
- 질의응답 API
- 문서 관리 API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import EnterpriseRAG
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

# FastAPI 앱
app = FastAPI(title="Enterprise RAG API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 시스템 초기화
rag = EnterpriseRAG(persist_directory="./chroma_db")

# Request Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Enterprise RAG API", "status": "running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """문서 업로드 및 임베딩"""
    try:
        # 파일 저장
        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # RAG에 추가
        num_chunks = rag.add_documents([file_path])
        
        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "chunks": num_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """질의응답"""
    try:
        result = rag.query(request.question)
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_database():
    """데이터베이스 초기화"""
    try:
        rag.clear_database()
        return {"message": "Database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 3. 실행 방법

```bash
# 1. 환경 변수 설정
echo "OPENAI_API_KEY=your-key" > .env

# 2. 서버 실행
cd enterprise_rag
uvicorn main:app --reload

# 3. API 테스트
# 문서 업로드
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sample.pdf"

# 질의
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "이자율은 얼마인가요?"}'
```

**체크포인트:**
- [ ] RAG 파이프라인 완성
- [ ] FastAPI 서버 구축
- [ ] 문서 업로드 기능
- [ ] 질의응답 정확도 90%+
- [ ] 출처 추적 (Citation)

---

## ✅ Week 4 완료 체크리스트

### 프로젝트 완성도
- [ ] Project 8: LangChain 챗봇 ✅
- [ ] Project 9: Agent (도구 활용) ✅
- [ ] Project 10: RAG 시스템 완성 ✅

### 기술 습득
- [ ] LangChain 완벽 숙련
- [ ] Vector Database 활용
- [ ] FastAPI 서버 구축
- [ ] RAG 파이프라인

### GitHub
- [ ] RAG 프로젝트 커밋
- [ ] API 문서 작성
- [ ] 사용 가이드 작성

### 다음 주 준비
- [ ] Docker 설치 확인
- [ ] MLflow 개념 학습
- [ ] 최종 프로젝트 기획

## 📊 학습 시간 기록

| 일자 | 활동 | 시간 | 완료 |
|------|------|------|------|
| Day 22 | LangChain 기초 | 8h | [ ] |
| Day 23 | 챗봇 + Agent | 8h | [ ] |
| Day 24 | Agent 고도화 | 8h | [ ] |
| Day 25 | Vector DB + 임베딩 | 8h | [ ] |
| Day 26 | RAG 파이프라인 | 8h | [ ] |
| Day 27 | FastAPI 서버 | 8h | [ ] |
| Day 28 | 통합 테스트 | 8h | [ ] |

---

**Week 4 완료 후 → Week 5 (통합 프로젝트 + MLOps)로 진행**
