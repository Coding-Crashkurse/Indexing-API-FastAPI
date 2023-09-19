import os
import logging
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from enum import Enum
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

CONNECTION_STRING = "postgresql+psycopg2://admin:admin@postgres:5432/vectordb"
COLLECTION_NAME = "vectordb"
namespace = f"pgvector/{COLLECTION_NAME}"

record_manager = SQLRecordManager(namespace, db_url=CONNECTION_STRING)
record_manager.create_schema()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRequest(BaseModel):
    page_content: str
    metadata: dict

class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)
vectorstore = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

prompt_template = """
You are a helpful assistant for our restaurant.

{context}

Question: {question}
Answer here:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/index")
async def index_documents(docs_request: list[DocumentRequest], cleanup: CleanupMethod = CleanupMethod.incremental) -> dict:
    documents = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in docs_request
    ]

    result = index(
        documents,
        record_manager,
        vectorstore,
        cleanup=cleanup.value,
        source_id_key="source",
    )

    return result

@app.post("/question")
async def ai_service(question: str) -> dict:
    result = qa.run(query=question)
    return {"result": result}
