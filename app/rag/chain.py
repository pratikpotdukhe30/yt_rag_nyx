from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from app.rag.llm import get_llm
from app.rag.prompts import RAG_PROMPT
from app.rag.vectorstore import get_vector_store


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_docs(docs: list[Document]) -> str:
    # Each chunk prefixed with its timestamp → the LLM can cite [mm:ss]
    return "\n\n".join(f"[{_fmt_time(d.metadata['start'])}] {d.page_content}" for d in docs)


def build_rag_chain(video_id: str) -> Runnable:
    """Input: question (str). Output: {"question", "docs", "answer"}; answer is None if nothing relevant."""
    retriever = get_vector_store().as_retriever(video_id)

    # Sub-chain that actually calls the LLM
    generate = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))  # add "context" key
        | RAG_PROMPT
        | get_llm()                 # OpenAI, auto-falls back to Groq
        | StrOutputParser()         # AIMessage → plain string
    )

    # LCEL routing: a RunnableLambda that RETURNS a Runnable → LangChain invokes it with the same input
    def route(inputs: dict) -> Runnable:
        if not inputs["docs"]:
            return RunnableLambda(lambda x: {**x, "answer": None})   # no relevant chunks → skip LLM
        return RunnablePassthrough.assign(answer=generate)

    return (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())  # retrieve + keep question
        | RunnableLambda(route)
    )