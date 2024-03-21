import os
import streamlit as st
import pdfplumber
import chromadb
import tempfile
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pdf2image import convert_from_path 

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


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

import os
import tempfile

def text_to_chromadb(file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
        f.write(file.getvalue())

    notas, extension = os.path.splitext(file.name)

    def load_pdf(file_path):
        from langchain.document_loaders import PyPDFLoader
        from PyPDF2 import PdfFileReader
        import pytesseract
        from PIL import Image
        import io

        print(f"Loading {file_path}")
        loader = PyPDFLoader(file_path)
        try:
            text = loader.load()
        except Exception as e:
            pdf_reader = PdfFileReader(open(file_path, 'rb'))
        text = ""
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            try:
                image_blob = page.extractText()
                image = Image.open(io.BytesIO(image_blob))
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text
            except Exception as e:
                print(f"Failed to perform OCR on page {page_num}: {e}")
        if not text:
            raise ValueError("Failed to extract text using PyPDF2 and OCR.")
        
        return loader.load()
    
    def load_docx(file_path):
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading {file_path}")
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def load_txt(file_path):
        from langchain.document_loaders import TextLoader
        print(f"Loading {file_path}")
        loader = TextLoader(file_path)
        return loader.load()

    def load_pptx(file_path):
        from langchain.document_loaders import UnstructuredPowerPointLoader
        print(f"Loading {file_path}")
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()
    
    def load_image(file_path):
        from langchain.document_loaders.image import UnstructuredImageLoader
        print(f"Loading {file_path}")
        loader = UnstructuredImageLoader(file_path)
        return loader.load()

    if extension == ".pdf":
        text = load_pdf(temp_filepath)
    elif extension == ".docx":
        text = load_docx(temp_filepath)
    elif extension == ".txt":
        text = load_txt(temp_filepath)
    elif extension == ".pptx":
        text = load_pptx(temp_filepath)
    else:
        # Handle other file formats or raise an error
        raise ValueError("Unsupported file format")

    with st.spinner(f'Creating embedding for file: {file.name}'):
        create_embeddings(file.name, text)
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
