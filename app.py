import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

st.set_page_config(page_title="Ask My Resume", page_icon="🧠")
st.title("🧠 Ask My Resume")
st.caption("AI-powered portfolio assistant — ask me anything about Giridhera")

# Cache so it doesn't reload on every interaction
@st.cache_resource
def build_rag_chain():
    # Load your resume PDF (place it in the repo as resume.pdf)
    loader = PyPDFLoader("resume.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    system_prompt = (
        "You are a professional assistant representing Giridhera Ramanan S, "
        "a final-year AI & Data Science student. Answer questions about his "
        "skills, projects, experience, and background using only the resume context. "
        "Be confident, concise, and professional. If the answer isn't in the resume, "
        "say 'That information isn't on the resume — feel free to reach out at "
        "giridheran007@gmail.com'\n\nContext: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Starter suggestions
    st.session_state.suggestions = [
        "What projects has Giridhera built?",
        "Does he have experience with LLMs?",
        "What is his tech stack?",
        "Tell me about his internship experience"
    ]

# Show suggestion buttons on first load
if not st.session_state.messages:
    st.write("**Try asking:**")
    cols = st.columns(2)
    for i, suggestion in enumerate(st.session_state.suggestions):
        if cols[i % 2].button(suggestion):
            st.session_state.messages.append(
                {"role": "user", "content": suggestion}
            )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask anything about Giridhera..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = build_rag_chain()
            result = chain.invoke({"input": question})
            answer = result["answer"]
            st.write(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )