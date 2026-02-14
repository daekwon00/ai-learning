# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Phase 기반 AI/ML 학습 포트폴리오. 별도 `rag-project`(Next.js + Supabase pgvector)에서 RAG를 이미 구현했으며, 여기서는 **Python 생태계**로 전환 학습.

| Phase | 디렉토리 | 상태 |
|-------|---------|------|
| 0 | `python-ml-basics/` | 완료 |
| 1 | `langchain-rag/` | 완료 |
| 2 | `ai-agent/` | 미착수 |
| 3 | `pytorch-bert/` | 미착수 |
| 4 | `deployment/` | 미착수 |

## Environment

```bash
# Conda 환경 (항상 이것 사용)
conda activate ai-dev    # Python 3.11

# Conda 초기화 (Bash tool에서)
eval "$('/opt/homebrew/Caskroom/miniforge/base/bin/conda' 'shell.bash' 'hook')" && conda activate ai-dev
```

- **Git remote**: SSH (`git@github.com:daekwon00/ai-learning.git`)
- **OpenAI API Key**: `langchain-rag/.env` (dotenv로 로드)
- **M4 Mac**: HuggingFace 임베딩에 `device="mps"` 사용

## Commands

```bash
# langchain-rag 모듈 테스트
python langchain-rag/modules/loader.py ./langchain-rag/data
python langchain-rag/modules/chunker.py
python langchain-rag/modules/vectorstore.py
python langchain-rag/modules/chain.py

# 노트북 실행
jupyter lab langchain-rag/rag_pipeline.ipynb

# 노트북 CLI 실행 (검증용)
jupyter nbconvert --to notebook --execute langchain-rag/rag_pipeline.ipynb --ExecutePreprocessor.timeout=180
```

## Architecture: langchain-rag

모듈식 RAG 파이프라인. 각 모듈은 독립 실행 가능 (`if __name__ == "__main__"`).

```
Documents → loader.py → chunker.py → vectorstore.py → chain.py → Answer
               ↓              ↓              ↓              ↓
          PyPDFLoader   500자/100자      ChromaDB      GPT-4o
          TextLoader    오버랩         HuggingFace    출처 표시
                        한국어 경계    all-MiniLM-L6-v2
```

- **chain.py**: `langchain_classic.chains`의 `create_retrieval_chain`, `create_history_aware_retriever` 사용 (LangChain 1.x)
- **vectorstore.py**: `langchain_chroma.Chroma` 사용 (구 `langchain.vectorstores.Chroma` 아님)
- `ConversationalRetrievalChain`은 deprecated → `create_history_aware_retriever` + `create_retrieval_chain` 조합

## LangChain 1.x 주의사항

```python
# 올바른 import
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# __version__ 접근 시 (일부 패키지에 없음)
from importlib.metadata import version
version('langchain-chroma')  # langchain_chroma.__version__은 없음
```

## Conventions

- 한국어로 질문하면 한국어로 답변
- 각 Phase 디렉토리에 `README.md`, `summary.md`, `requirements.txt` 포함
- 노트북은 실행 결과 포함 상태로 커밋
- `chroma_db/`, `.env`, `__pycache__/`는 gitignore 대상
