# LangChain RAG 학습 정리

## TypeScript(rag-project) vs Python(langchain-rag) 비교

### 파일 대응표

| rag-project 파일 | langchain-rag 파일 | 역할 |
|---|---|---|
| `app/api/ingest/route.ts` | `modules/loader.py` + `chunker.py` | 문서 수집 파이프라인 |
| `lib/chunker.ts` | `modules/chunker.py` | 500자/100자 오버랩, 문장 경계 분할 |
| `lib/ai/embedding.ts` | `modules/vectorstore.py` | 임베딩 생성 + 유사도 검색 |
| `lib/db/index.ts` + `schema.ts` | `modules/vectorstore.py` | DB 저장/검색 (pgvector → ChromaDB) |
| `app/api/chat/route.ts` | `modules/chain.py` | 시스템 프롬프트 + LLM 응답 생성 |

### 핵심 코드 대응

#### 1. 문서 청킹

**TypeScript (직접 구현)**
```typescript
// lib/chunker.ts
export function chunkText(text: string, chunkSize = 500, overlap = 100) {
  // 문장 경계 감지: ". ", ".\n", "다. ", "다.\n"
  // 30% 이상에서만 경계 사용
}
```

**Python (LangChain 추상화)**
```python
# modules/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "다.\n", "다. ", ".\n", ". ", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
```

#### 2. 임베딩 + 벡터 저장

**TypeScript (OpenAI + Supabase pgvector)**
```typescript
// lib/ai/embedding.ts
const embedding = await openai.embeddings.create({
  model: "text-embedding-3-small",
  input: text,
});

// lib/db/index.ts (SQL 기반 유사도 검색)
const results = await db.execute(sql`
  SELECT *, 1 - (embedding <=> ${queryEmbedding}) as similarity
  FROM embeddings ORDER BY similarity DESC LIMIT 5
`);
```

**Python (HuggingFace + ChromaDB)**
```python
# modules/vectorstore.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embedding)
results = vectorstore.similarity_search(query, k=5)
```

#### 3. QA 체인

**TypeScript (직접 프롬프트 조합)**
```typescript
// app/api/chat/route.ts
const systemPrompt = `You are a document-based QA assistant...
Context: ${relevantDocs.map(d => d.content).join('\n')}`;

const result = await streamText({
  model: openai("gpt-4o"),
  system: systemPrompt,
  messages,
});
```

**Python (LangChain 체인)**
```python
# modules/chain.py
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "문서 기반 QA 어시스턴트... {context}"),
    ("human", "{input}"),
])
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, qa_prompt))
result = chain.invoke({"input": question})
```

### 주요 차이점 정리

| 관점 | TypeScript (rag-project) | Python (langchain-rag) |
|------|------------------------|----------------------|
| **추상화** | 각 단계 직접 구현 | LangChain이 파이프라인 추상화 |
| **임베딩 비용** | OpenAI API ($) | HuggingFace 로컬 (무료) |
| **벡터 DB** | pgvector (SQL, 클라우드) | ChromaDB (파일, 로컬) |
| **대화형** | 미지원 | history_aware_retriever |
| **배포** | Vercel 서버리스 | 로컬/노트북 |
| **장점** | 프로덕션 즉시 배포 | 빠른 프로토타이핑, 무료 실험 |

### 학습 포인트

1. **LangChain LCEL**: LangChain 1.x에서는 `langchain_classic`의 `create_retrieval_chain` 사용. 과거 `ConversationalRetrievalChain`은 deprecated.
2. **HuggingFace 임베딩**: `all-MiniLM-L6-v2`는 영어 최적화. 한국어에서는 다국어 모델 (`multilingual-e5-large`) 권장.
3. **ChromaDB**: 로컬 파일 기반으로 별도 서버 불필요. 프로토타이핑에 적합하지만 프로덕션에서는 pgvector/Pinecone 등 사용.
4. **MPS 가속**: M4 Mac에서 `device="mps"` 설정으로 HuggingFace 임베딩 가속 가능.
