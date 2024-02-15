import streamlit as st
import chromadb
import openai
import os 
from fastapi import FastAPI
from langchain.agents import AgentExecutor,tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.llms import openai
from decouple import config
from langchain import PromptTemplate
from typing import Optional, Union, List
from pydantic import BaseModel, ValidationError
#from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
from pdfminer.high_level import extract_pages
from pdfminer.high_level import extract_text
import pytesseract
import shutil


### PATHS ###
# Para instalar teseract deberas hacer los pasos que se realizan en el siguiente link: https://www.youtube.com/watch?v=3Q1gTDXzGnU&t=12s
# si tienes ubuntu ejecuta en la terminal "sudo apt-get install tesseract-ocr"

# Al finalizar utiliza el comando que se encuenta en /Conseguir_Rutas.ipynb para verificar
# que la ruta esta correctamente implementada, al finalizar, solo pega la ruta en el codigo que se encuentra abajo

#pytesseract.pytesseract.tesseract_cmd = os.environ["UBI_TESSERACT"]

### CARGAR DOCUMENTOS ###


#Emulando 'switch' en Python por medio de diccionarios y de este modo se agiliza el proceso

class DocumentInfo(BaseModel):
    notas: Union[str, List[str]]
    extension: str
    content: Optional[str] = None 

def load_document(file):
    
    import os
    
    notas, extension = os.path.splitext(file)

    def load_pdf(file):
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading {file}")
        loader = PyPDFLoader(file)
        return loader.load()

    def load_docx(file):
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading {file}")
        loader = Docx2txtLoader(file)
        return loader.load()

    def load_txt(file):
        from langchain.document_loaders import TextLoader
        print(f"Loading {file}")
        loader = TextLoader(file)
        return loader.load()

    def load_pptx(file):
        
        from langchain.document_loaders import UnstructuredPowerPointLoader
        print(f"Loading {file}")
        loader = UnstructuredPowerPointLoader(file)
        return loader.load()
    
    def load_image(file):
        
        from langchain.document_loaders.image import UnstructuredImageLoader
        print(f"Loading {file}")
        loader = UnstructuredImageLoader(file)
        return loader.load()

     ### Formato válido
    def default_loader(file):
        print(f"Documento '{file}' no soportado o tipo de archivo '{extension}' no reconocido")
        return DocumentInfo(
        notas, Union[str, List[str]],
        extension, str,
        content= Optional[str]
        )
		
  # Creamos un diccionario que asocia el tipo de documento con la respuesta que dará a la hora de cargar el documento	
    switcher = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".pptx": load_pptx,
        ".jpg": load_image,
        ".png": load_image,
    }
    
    loader_function = switcher.get(extension, default_loader)
    document_info = loader_function(file)
    

    print (document_info)
    return document_info    


    


### CARGAR CHUNKS ###


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks


### CARGAR EMBEDDINGS ###


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


### SPEAKING ###

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
def ask_and_get_answer(vector_store, q, k=3):


    llm = ChatOpenAI(model="text-embedding-ada-002", temperature=0.7)

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    answer = chain.run(q)
    return answer


### calcular costos del embedding ###


def calculate_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("babbage-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]


def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0.7)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))

    return result, chat_history


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)

    st.image("public/s4b_logo.png")
    st.subheader(
        "¡Hola usuario!, mi nombre es Lay Sphere🤖, fui diseñada por silent4business para analizar los documentos que me proporciones y me preguntes sobre ellos y te responda con mucho gusto 😀, dicho lo anterior, porfavor..."
    )
    background_image_path = os.path.join(os.getcwd(), "fondo_layla_sphere.jpg")
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("{background_image_path}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        api_key = os.environ["OPENAI_API_KEY"]
        #api_key = st.text_input("OPENAI_API_KEY:", type="password")
        #if api_key:
        #    os.environ["OPENAI_API_KEY"] = api_key

        uploaded_file = st.file_uploader(
            "Upload a file:", type=["pdf", "docx", "txt", "pptx", "jpg", "png"]
        )
        chunk_size = st.number_input(
            "Chunk size:",
            min_value=100,
            max_value=2048,
            value=512,
            on_change=clear_history,
        )
        k = st.number_input(
            "k", min_value=1, max_value=20, value=3, on_change=clear_history
        )
        add_data = st.button("Add Data", on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner("Reading, chunking and embedding file ..."):
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)

            data = load_document(file_name)
            chunks = chunk_data(data, chunk_size=chunk_size)
            st.write(f"Chunk size: {chunk_size}, Chunks: {len(chunks)}")

            tokens, embedding_cost = calculate_embedding_cost(chunks)
            st.write(f"Embedding cost: ${embedding_cost:.4f}")

            vector_store = create_embeddings(chunks)
            st.session_state.vs = vector_store
            st.session_state.chat_history = []

            st.session_state.vs = vector_store
            st.success("File uploaded, chunked and embedded sucessfully")

    q = st.text_input("Haz una pregunta acerca del contenido de tu archivo 😀")
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            st.write(f"k: {k}")
            result, st.session_state.chat_history = ask_with_memory(
                vector_store, q, st.session_state.chat_history
            )
            answer = result["answer"]
            st.text_area("LLM Answer: ", value=answer)

            st.divider()
            if "history" not in st.session_state:
                st.session_state.history = ""
            value = f"Q: {q} \nA: {answer}"
            st.session_state.history = (
                f'{value} \n {"-" * 100} \n {st.session_state.history}'
            )
            h = st.session_state.history
           
 
                
            

load_dotenv(find_dotenv(),override=True)
def OPENAI_API_KEY(self):
    return os.environ.get("OPENAI_API_KEY", "")

