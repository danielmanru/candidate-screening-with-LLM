from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_pdf_text(pdf_docs):
    loader = PyPDFLoader(pdf_docs)
    document = loader.load()
    return document[0].page_content

job_desc_docs = get_pdf_text("data/job-desc.pdf")
score_rubric_docs = get_pdf_text("data/scoring-rubric.pdf")
db_location = "./cv_screening_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name= "job-data",
    persist_directory= db_location,
    embedding_function= embeddings
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

job_chunks = text_splitter.split_text(job_desc_docs)
rubric_chunks = text_splitter.split_text(score_rubric_docs)

doc_count = len(vector_store.get()["ids"])
if doc_count == 0:
    documents = []
    for i, chunk in enumerate(job_chunks):
        documents.append(Document(
            page_content=chunk,
            metadata={"type": "job description", "chunk": i}
        ))

    for i, chunk in enumerate(rubric_chunks):
        documents.append(Document(
            page_content=chunk,
            metadata={"type": "score rubric", "chunk": i}
        ))

    vector_store.add_documents(documents)

retriever = vector_store.as_retriever(
    search_kwargs= {"k": 1}
)