import os
import io
import streamlit as st

import chromadb
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

FILE_LIST = "archivos.txt"
INDEX_NAME = 'taller'

# Corrected Chroma server hostname and port
chroma_client = chromadb.HttpClient(host='chroma-lay-db', port=8000)

def save_name_files(path, new_files):
    old_files = load_name_files(path)
    with open(path, "a") as file:
        for item in new_files:
            if item not in old_files:
                file.write(item + "\n")
                old_files.append(item)
    return old_files

def load_name_files(path):
    archivos = []
    with open(path, "r") as file:
        for line in file:
            archivos.append(line.strip())
    return archivos

def clean_files(path):
    with open(path, "w") as file:
        pass
    chroma_client.delete_collection(name=INDEX_NAME)
    collection = chroma_client.create_collection(name=INDEX_NAME)
    return True

def is_scanned_pdf(pdf_path):
    """
    Determina si un archivo PDF contiene hojas escaneadas
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        if page.get_text() == "":
            return True  # Si alguna página no tiene texto, consideramos que es escaneada
    return False

def extract_text_from_image(image):
    """
    Extrae texto de una imagen utilizando OCR
    """
    return pytesseract.image_to_string(image)

def text_to_chromadb(pdf):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf.getvalue())

    if is_scanned_pdf(temp_filepath):
        st.write("El archivo PDF contiene hojas escaneadas.")
        extracted_text = ""
        doc = fitz.open(temp_filepath)
        for page_number, page in enumerate(doc):
            image_list = page.get_images()
            if image_list:
                st.write(f"Extrayendo texto de las imágenes en la página {page_number + 1}")
                for image_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    extracted_text += extract_text_from_image(image)
        st.write("Texto extraído de las imágenes:", extracted_text)
        create_embeddings_text(pdf.name, extracted_text) 
    else:
        st.write("El archivo PDF no contiene hojas escaneadas.")
        loader = PyPDFLoader(temp_filepath)
        scanned_document = loader.load()
        create_embeddings(pdf.name, scanned_document)  # Llamar a create_embeddings con el documento
        
        return True


def create_embeddings_text(file_name, text):
    print(f"Creating embeddings for the file: {file_name}")

    # Inicializar RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    # Dividir el texto en fragmentos
    chunks = [text]

    # Inicializar HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Crear embeddings para el texto y almacenarlos en Chroma
    Chroma.from_documents(
        chunks,
        embeddings,
        client=chroma_client,
        collection_name=INDEX_NAME
    )

    return True



def create_embeddings(file_name, text):
    print(f"Creating embeddings for the file: {file_name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )        
    
    chunks = text_splitter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    
    Chroma.from_documents(
        chunks,
        embeddings,   
        client=chroma_client,
        collection_name=INDEX_NAME)
        
    return True
