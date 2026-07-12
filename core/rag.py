from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vectorstore import build_vectorstore, load_vector_store, get_retriever

load_dotenv()

SYSTEM_PROMPT = """You are Rewise, a friendly, sharp, knowledgeable AI assistant that helps people explore and understand recorded audio or video content — meetings, lectures, podcasts, interviews, YouTube videos, or any other recording.
You answer the user's question using only the transcript context provided below. Be precise, concise, and helpful, and when useful, quote or reference the relevant part of the recording.
If the answer is not found in the context, say:
"I could not find this in the recording."
The context will be provided under: {context}
"""


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )


def format_doc(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def _build_chain_from_vector_store(vector_store):
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_doc),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str, session_id: str):
    vector_store = build_vectorstore(transcript, session_id)
    return _build_chain_from_vector_store(vector_store)


def load_rag_chain(session_id: str):
    vector_store = load_vector_store(session_id)
    return _build_chain_from_vector_store(vector_store)


def ask_question(ragchain, question: str) -> str:
    print(f"ask your question : {question}")
    answer = ragchain.invoke(question)
    print(f"answer : {answer}")
    return answer