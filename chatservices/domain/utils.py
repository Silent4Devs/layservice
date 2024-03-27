import os
import io
import streamlit as st

import chromadb
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredImageLoader
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

def text_to_chromadb(file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
        f.write(file.getvalue())

    if file.type == "application/pdf":
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

            new_pdf_path = os.path.join(temp_dir.name, "text_extracted.pdf")
            c = canvas.Canvas(new_pdf_path, pagesize=letter)
            c.drawString(100, 750, extracted_text)
            c.save()

            loader = PyPDFLoader(new_pdf_path)
            text = loader.load()
            create_embeddings(file.name, text)


            return True
        
        elif file.type == "text/plain":
            # Manejar archivos de texto (txt)
            with open(temp_filepath, "r", encoding="utf-8") as f:
                text = f.read()
            create_embeddings(file.name, text)
            return True
        
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Manejar archivos de Word (docx)
            from langchain.document_loaders import word_document
            doc = word_document(temp_filepath)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            create_embeddings(file.name, text)
            return True

        elif file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            # Manejar archivos de PowerPoint (pptx)
            from pptx import Presentation
            prs = Presentation(temp_filepath)
            text = "\n".join([slide.text for slide in prs.slides])
            create_embeddings(file.name, text)
            return True

        elif st.write("El archivo PDF no contiene hojas escaneadas."):
            loader = PyPDFLoader(temp_filepath)
            text = loader.load()
            create_embeddings(file.name, text)

            return True
        else:
            st.write(f"Tipo de archivo no compatible: {file.type}")
            return False
    

    

    






def create_embeddings(file_name, text):
        st.write(f"Creating embeddings for the file: {file_name}")

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

def log_interaction(question, answer):
    """
    Registra la pregunta realizada por el usuario y su respuesta asociada en un archivo de registro.
    """
    with open("registro_preguntas.txt", "a") as file:
        file.write(f"Pregunta: {question}\n")
        file.write(f"Respuesta: {answer}\n\n")
        

def buscar_respuesta(pregunta):
    with open("registro_preguntas.txt", "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):  # Buscar en el archivo línea por línea
            if lines[i].strip() == f"Pregunta: {pregunta}":  # Si se encuentra la pregunta
                return lines[i + 1].strip()  # Devolver la respuesta asociada
    return None  # Si la pregunta no se encuentra en el archivo

