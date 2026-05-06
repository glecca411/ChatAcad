import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA

# 1. Configuración de la página
st.set_page_config(page_title="Asistente Normativo Universitario", page_icon="🎓")

st.title("🤖 Asistente Virtual de Reglamentos Académicos")
st.markdown("""
Esta aplicación permite consultar dudas sobre la normativa universitaria utilizando Inteligencia Artificial.
---
""")

# 2. Gestión de la API Key de forma segura
# En Streamlit Cloud, esto se configura en "Advanced Settings > Secrets"
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Falta la configuración de la API Key. Por favor, añádela en los Secrets de Streamlit.")
    st.stop()

# 3. Función para procesar el PDF y crear la base de datos de conocimientos
@st.cache_resource
def procesar_reglamento(archivo_subido):
    # Guardar el archivo temporalmente
    with open("documento_temp.pdf", "wb") as f:
        f.write(archivo_subido.getbuffer())
    
    # Cargar y dividir el texto
    loader = PyPDFLoader("documento_temp.pdf")
    documentos = loader.load()
    
    # Dividimos el reglamento en fragmentos para que la búsqueda sea precisa
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        length_function=len
    )
    fragmentos = text_splitter.split_documents(documentos)
    
    # Crear vectores (representación numérica del texto)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings)
    
    return vectorstore

# 4. Interfaz de carga
archivo = st.file_uploader("Sube el reglamento oficial en formato PDF", type="pdf")

if archivo:
    with st.spinner("Analizando el reglamento... esto tardará solo unos segundos."):
        base_conocimiento = procesar_reglamento(archivo)
        
        # Configuración del modelo de lenguaje
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        
        # Crear la cadena de respuesta (RAG)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=base_conocimiento.as_retriever(search_kwargs={"k": 3})
        )

    st.success("¡Reglamento cargado con éxito! Ya puedes hacer tus consultas.")

    # 5. Historial del Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes previos
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de usuario
    if pregunta := st.chat_input("Ejemplo: ¿Cuántas veces puedo desaprobar un curso?"):
        # Agregar pregunta al historial
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Consultando la normativa..."):
                # Instrucción estricta para evitar que la IA invente reglas
                prompt_estricto = f"""
                Eres un asistente legal académico experto. 
                Usa ÚNICAMENTE los fragmentos del reglamento proporcionado para responder.
                Si la respuesta no aparece de forma explícita, responde: 
                'Lo siento, no encontré esa información en el reglamento. Por favor, contacta con la oficina de Registro Académico.'
                
                Pregunta del estudiante: {pregunta}
                """
                respuesta = qa_chain.invoke(prompt_estricto)
                st.markdown(respuesta["result"])
                st.session_state.messages.append({"role": "assistant", "content": respuesta["result"]})
