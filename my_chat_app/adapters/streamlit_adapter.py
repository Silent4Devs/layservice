# adapters/streamlit_adapter.py
import streamlit as st
from application.use_cases import upload_document_interface, answer_question_interface
from infrastructure.database_repository import DatabaseRepository

def main():
    st.markdown("<h1 style='font-size:24px;'>¡Hola usuario!, mi nombre es Lay Sphere🤖, fui diseñada por silent4business para analizar los documentos que me proporciones y me preguntes sobre ellos y te responda con mucho gusto 😀, dicho lo anterior, porfavor...</h1>", unsafe_allow_html=True)

    # Interfaz para cargar el documento desde el sistema local
    uploaded_file = st.file_uploader("Cargar documento PDF", type="pdf")
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()  # Obtener los bytes del archivo
        document_info = upload_document_interface(bytes_data)
        st.write("Documento cargado exitosamente:", document_info)

    # Interfaz para hacer preguntas personalizadas
    question = st.text_input("Haz una pregunta personalizada:")
    if question:
        # Asumiendo que ya se ha cargado un documento previamente
        if 'document_info' in locals():
            database_repo = DatabaseRepository()
            answer = answer_question_interface(question, document_info, database_repo)
            st.write("Respuesta:", answer)
        else:
            st.write("Primero carga un documento antes de hacer preguntas personalizadas.")

if __name__ == "__main__":
    main()
