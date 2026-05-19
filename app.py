from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://weirdowrench.github.io"],
    allow_methods=["GET", "POST", "OPTIONS"],  
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

resume_text = ""

@app.on_event("startup")
async def startup():
    global resume_text, chain

    # Load and extract full resume text
    loader = PyPDFLoader("resume.pdf")
    pages = loader.load()
    resume_text = "\n\n".join(page.page_content for page in pages)

    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    system_prompt = (
        "You are a professional assistant representing Giridhera Ramanan S, "
        "a final-year AI & Data Science student. Answer questions about his "
        "skills, projects, experience, and background using ONLY the resume below. "
        "Be confident, concise and professional. If the answer isn't in the resume "
        "say: 'That information isn't on the resume — feel free to reach out at "
        "giridheran007@gmail.com'\n\nResume:\n{resume}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

@app.post("/ask")
async def ask(body: Question):
    answer = chain.invoke({"resume": resume_text, "input": body.question})
    return {"answer": answer}

@app.get("/")
async def root():
    return {"status": "ok", "message": "Resume RAG API is running"}