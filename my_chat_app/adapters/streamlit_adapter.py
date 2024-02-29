# adapters/streamlit_adapter.py

import streamlit as st
from application.use_cases import upload_document, answer_question
from domain.document import DocumentInfo
from domain.chat_interaction import ChatAnswer

def upload_document_interface():
    uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt", "pptx", "jpg", "png"])
    document_info = None  # Inicializar como None en caso de que no se cargue ningún archivo
    if uploaded_file:
        with st.spinner("Reading, chunking and embedding file ..."):
            bytes_data = uploaded_file.read()
            document_info = upload_document(bytes_data)
        st.success("File uploaded, chunked and embedded successfully")
    return document_info

def main():
    st.image("my_chat_app/public/s4b_logo.png")
    st.subheader("¡Hola usuario!, mi nombre es Lay Sphere🤖, fui diseñada por silent4business para analizar los documentos que me proporciones y me preguntes sobre ellos y te responda con mucho gusto 😀, dicho lo anterior, porfavor...")
    
    # Agregar el input para la pregunta antes de cargar el archivo
    question = st.text_input("Haz una pregunta acerca del contenido de tu archivo 😀")

    # Cambiar el nombre de la función para reflejar mejor su propósito
    document_info = upload_document_interface()
    
    # Luego de cargar el archivo
    if document_info:
        # Cambiar el nombre de la función para reflejar mejor su propósito
        if question:
            answer = answer_question(document_info, question)
            st.text_area("LLM Answer: ", value=answer.answer)

if __name__ == "__main__":
    main()
