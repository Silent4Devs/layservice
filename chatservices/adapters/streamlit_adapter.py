import streamlit as st
import os
from domain.utils import *
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv, find_dotenv
from domain.utils import log_interaction


load_dotenv(find_dotenv(), override=True)
def run():

    st.set_page_config('preguntaDOC')
    st.header("¡Hola usuario!, mi nombre es Lay Sphere🤖, fui diseñada por silent4business para analizar los documentos que me proporciones y me preguntes sobre ellos y te responda con mucho gusto 😀, dicho lo anterior, porfavor...")


    with st.sidebar:
        
        archivos = load_name_files(FILE_LIST)
        files_uploaded = st.file_uploader(
            "Carga tu archivo",
            type="pdf",
            accept_multiple_files=True
            )
        
        if st.button('Procesar'):
            for pdf in files_uploaded:
                if pdf is not None and pdf.name not in archivos:
                    archivos.append(pdf.name)
                    text_to_chromadb(pdf)

            archivos = save_name_files(FILE_LIST, archivos)

        if len(archivos) > 0:
            st.write("Archivos cargados:")
            lista_documentos = st.empty()
            with lista_documentos.container():
                for arch in archivos:
                    st.write(arch)
                if st.button('Borrar documentos'):
                    archivos = []
                    clean_files(FILE_LIST)
                    lista_documentos.empty()

    if archivos:
        user_question = st.text_input("Pregunta:")
        if user_question:
        # Resto de tu código para procesar la pregunta
            os.environ["OPENAI_API_KEY"]
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            
            vstore = Chroma(client=chroma_client,
                            collection_name=INDEX_NAME,
                            embedding_function=embeddings)

            docs = vstore.similarity_search(user_question, 3)
            llm = ChatOpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            respuesta = chain.run(input_documents=docs, question=user_question)

            respuesta_guardada = buscar_respuesta(user_question)
            
            if respuesta_guardada:
                st.write(respuesta_guardada)
            else:
                log_interaction(user_question, respuesta)  # Registrar la pregunta y la respuesta

            st.write(respuesta)


    

if __name__ == "__main__":
    run()