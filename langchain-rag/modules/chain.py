"""QA 체인 모듈

rag-project 대응: app/api/chat/route.ts
- 기본 QA: create_retrieval_chain + create_stuff_documents_chain
- 대화형 RAG: create_history_aware_retriever (멀티턴 대화 지원)
- 시스템 프롬프트: rag-project의 출처 표시 패턴 유지
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# rag-project의 시스템 프롬프트와 유사한 패턴
QA_SYSTEM_PROMPT = """당신은 문서 기반 Q&A 어시스턴트입니다.
아래 제공된 컨텍스트를 기반으로 질문에 답변하세요.

규칙:
1. 반드시 제공된 컨텍스트에 기반하여 답변하세요.
2. 답변 후 참고한 문서 출처를 표시하세요.
3. 컨텍스트에서 답을 찾을 수 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답하세요.

컨텍스트:
{context}"""

# 대화형 RAG에서 이전 대화를 고려한 질문 재작성 프롬프트
CONTEXTUALIZE_PROMPT = """주어진 대화 기록과 최신 사용자 질문을 보고,
대화 기록의 맥락을 참조해야 하는 경우 독립적으로 이해할 수 있는 질문으로 재작성하세요.
질문에 답변하지 말고, 필요하면 재작성만 하세요. 그렇지 않으면 그대로 반환하세요."""


def _get_llm(model: str = "gpt-4o", temperature: float = 0) -> ChatOpenAI:
    """ChatOpenAI 인스턴스를 반환한다."""
    return ChatOpenAI(model=model, temperature=temperature)


def create_qa_chain(vectorstore: Chroma, model: str = "gpt-4o"):
    """기본 QA 체인을 생성한다.

    rag-project의 단일 질문-응답 패턴 대응.
    create_retrieval_chain으로 retriever + stuff_documents 결합.

    Args:
        vectorstore: Chroma 인스턴스
        model: OpenAI 모델명

    Returns:
        Runnable 체인 (input: {"input": str} → output: {"answer": str, "context": [Document]})
    """
    llm = _get_llm(model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def create_conversational_chain(vectorstore: Chroma, model: str = "gpt-4o"):
    """대화형 RAG 체인을 생성한다.

    rag-project에는 없는 확장 기능: 멀티턴 대화에서 이전 맥락을 고려.
    history_aware_retriever로 질문을 재작성한 후 검색.

    Args:
        vectorstore: Chroma 인스턴스
        model: OpenAI 모델명

    Returns:
        Runnable 체인 (input: {"input": str, "chat_history": list} → output: {"answer": str})
    """
    llm = _get_llm(model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Step 1: 대화 맥락을 고려한 질문 재작성
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Step 2: QA 체인
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def ask(chain, question: str, chat_history: list | None = None) -> dict:
    """체인에 질문하고 결과를 출력한다.

    Args:
        chain: QA 체인 또는 대화형 RAG 체인
        question: 질문
        chat_history: 이전 대화 기록 (대화형 RAG에서만 사용)

    Returns:
        {"answer": str, "context": list[Document], "chat_history": list}
    """
    input_data = {"input": question}
    if chat_history is not None:
        input_data["chat_history"] = chat_history
    else:
        chat_history = []

    result = chain.invoke(input_data)

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")
    print(f"A: {result['answer']}")

    if result.get("context"):
        print(f"\n📄 참조 문서 ({len(result['context'])}개):")
        sources = set()
        for doc in result["context"]:
            src = doc.metadata.get("source", "N/A")
            if src not in sources:
                sources.add(src)
                print(f"  - {src}")

    # 대화 기록 업데이트
    updated_history = chat_history + [
        HumanMessage(content=question),
        AIMessage(content=result["answer"]),
    ]

    return {
        "answer": result["answer"],
        "context": result.get("context", []),
        "chat_history": updated_history,
    }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    from vectorstore import get_vectorstore

    vs = get_vectorstore()
    chain = create_qa_chain(vs)

    ask(chain, "RAG란 무엇인가요?")
    ask(chain, "Next.js의 주요 특징을 설명해주세요.")
