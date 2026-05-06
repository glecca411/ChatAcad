import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Asistente Académico", page_icon="🎓")
st.title("🤖 Chat de Reglamentos Universitarios")

# Configuración de API Key desde los Secrets de Streamlit
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Configura la OPENAI_API_KEY en los Secrets de Streamlit.")
    st.stop()

@st.cache_resource
def procesar_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return vectorstore

archivo = st.file_uploader("Sube el reglamento (PDF)", type="pdf")

if archivo:
    vectorstore = procesar_pdf(archivo)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # gpt-3.5 es más económico y estable para pruebas
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("¿Qué dice el reglamento sobre...?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            respuesta = qa.invoke(prompt)
            st.markdown(respuesta["result"])
            st.session_state.messages.append({"role": "assistant", "content": respuesta["result"]})
