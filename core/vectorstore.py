from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_DIR = "vector_store_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    return embeddings


def _collection_name(session_id: str) -> str:
    return f"meeting_{session_id}"


def build_vectorstore(transcript: str, session_id: str) -> Chroma:
    print("Building vector store (❁´◡`❁)")

    embeddings = get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=_collection_name(session_id),
        persist_directory=CHROMA_DIR
    )

    return vector_store


def load_vector_store(session_id: str) -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=_collection_name(session_id),
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )