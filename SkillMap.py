import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
#  Page config
st.set_page_config(page_title="Resume Optimizing Mentor", page_icon="🧑‍🏫", layout="wide")

#  Session State init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 3. Sidebar for uploading PDF
st.sidebar.header("📄 Upload your Resume")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp_uploaded_file.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    st.session_state.vectorstore = vectorstore
    st.sidebar.success("✅ PDF uploaded and processed!")

else:
    st.sidebar.info("Upload a PDF to enable Analysis).")

#  Main Title
st.title("🧑‍🏫 Welcome to SkillMap — Personalized Resume Analyzer")

#  Chat input
input_text = st.chat_input("💬 Ask your question:")

if input_text:
    # Show user's message
    st.chat_message("user").markdown(input_text)
    st.session_state.messages.append({"role": "user", "content": input_text})

    # Initialize LLM
    llm = Ollama(model="gemma:2b")
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        input_key="question",    
        output_key="answer" 
    )

    #  Check if vectorstore is ready (PDF uploaded)
    if st.session_state.vectorstore is not None:
        retriever = st.session_state.vectorstore.as_retriever()

        # retrieve relevant docs
        docs = retriever.invoke(input_text)

        # manually build the context
        context = "\n\n".join(doc.page_content for doc in docs)

        # manually build final prompt
        final_prompt = f"""You are a resume optimizer assistant. Always reply helping user to improve resume, in proffessional teaching style like a HR of a MNC. 
Add helpful tips or suggestions with  remarks when appropriate.

Here is some context to help you answer:
{context}

Now, based on the context, answer the following question like a mentor helping build a strong resume:
{input_text}

If you don't know the answer, make up something impressive but mention you are you are not sure.

"""

        answer = llm.invoke(final_prompt)

    else:
        # fallback if no PDF
        answer = llm.invoke(input_text)

    # Show assistant's message
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
