from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os

def get_pdf_text(pdf_docs):
    loader = PyPDFLoader(pdf_docs)
    documents = loader.load()
    return documents[0].page_content

job_desc_docs = get_pdf_text("data/job-desc.pdf")
score_rubric_docs = get_pdf_text("data/scoring-rubric.pdf")
db_location = "./cv_screening_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name= "job-data",
    persist_directory= db_location,
    embedding_function= embeddings
)

doc_count = len(vector_store.get()["ids"])
if doc_count == 0:
    vector_store.add_documents([
        Document(page_content=job_desc_docs, metadata={"type": "job description"}),
        Document(page_content=score_rubric_docs, metadata={"type": "score rubric"}),
    ])

retriever = vector_store.as_retriever(
    search_kwargs= {"k": 1}
)