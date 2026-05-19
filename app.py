from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://weirdowrench.github.io"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.on_event("startup")
async def startup():
    global rag_chain
    loader = PyPDFLoader("resume.pdf")
    documents = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
    system_prompt = (
        "You are a professional assistant representing Giridhera Ramanan S, "
        "a final-year AI & Data Science student. Answer questions about his "
        "skills, projects, experience, and background using only the resume context. "
        "Be confident, concise and professional. If the answer isn't in the resume "
        "say: 'That information isn't on the resume — feel free to reach out at "
        "giridheran007@gmail.com'\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Pure LCEL chain — no langchain.chains needed
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

@app.post("/ask")
async def ask(body: Question):
    answer = rag_chain.invoke(body.question)
    return {"answer": answer}