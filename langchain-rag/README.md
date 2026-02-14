# Phase 1: Python LangChain RAG

> rag-project(Next.js/TypeScript) 경험을 Python/LangChain으로 전환

## 학습 목표

- LangChain + ChromaDB RAG 파이프라인 구축
- PyPDFLoader → RecursiveCharacterTextSplitter → ChromaDB
- create_retrieval_chain → history_aware_retriever (대화형 RAG)
- HuggingFace 로컬 임베딩 (all-MiniLM-L6-v2) 활용

## rag-project와의 차이

| 항목 | rag-project (완료) | 이번 Phase |
|------|-------------------|-----------|
| 언어 | TypeScript | Python |
| 프레임워크 | Next.js + Vercel AI SDK | LangChain |
| 벡터 DB | Supabase pgvector | ChromaDB (로컬) |
| 임베딩 | OpenAI text-embedding-3-small | HuggingFace all-MiniLM-L6-v2 |
| LLM | GPT-4o | GPT-4o (LangChain 래퍼) |
| 대화형 | 미지원 | history_aware_retriever |

## 디렉토리 구조

```
langchain-rag/
├── README.md
├── requirements.txt
├── .env                         # OPENAI_API_KEY (gitignore)
├── rag_pipeline.ipynb           # 메인 학습 노트북 (5섹션)
├── modules/
│   ├── __init__.py
│   ├── loader.py                # 문서 로더 (PDF + TXT)
│   ├── chunker.py               # RecursiveCharacterTextSplitter
│   ├── vectorstore.py           # ChromaDB + HuggingFace/OpenAI 임베딩
│   └── chain.py                 # QA 체인 + 대화형 RAG
├── data/                        # 테스트 문서 (rag-project에서 복사)
│   ├── vercel-ai-sdk-guide.pdf
│   ├── typescript-basics.txt
│   ├── nextjs-intro.txt
│   ├── rag-explanation.txt
│   └── supabase-guide.txt
├── chroma_db/                   # ChromaDB 로컬 저장소 (gitignore)
└── summary.md                   # TypeScript vs Python 비교
```

## 환경

```bash
conda activate ai-dev
pip install -r requirements.txt
```

## 실행

```bash
# 노트북 실행
jupyter lab rag_pipeline.ipynb

# 모듈별 테스트
python modules/loader.py ./data
python modules/chunker.py
python modules/vectorstore.py
python modules/chain.py
```

## 상태: 완료
