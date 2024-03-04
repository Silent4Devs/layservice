# adapters/streamlit_adapter.py
import streamlit as st
from application.use_cases import upload_document, answer_question
from domain.document import DocumentInfo
from domain.chat_interaction import ChatAnswer
import chromadb
import openai
from langchain.embeddings.openai import OpenAIEmbeddings



def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

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



def main():
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    background_image_path = os.path.join(os.getcwd(), "fondo_layla_sphere.jpg")
    st.image("my_chat_app/public/s4b_logo.png")
    st.subheader("¡Hola usuario!, mi nombre es Lay Sphere🤖, fui diseñada por silent4business para analizar los documentos que me proporciones y me preguntes sobre ellos y te responda con mucho gusto 😀, dicho lo anterior, porfavor...")
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
    with st.sidebar:   ########
        import os
        from dotenv import load_dotenv, find_dotenv

        load_dotenv(find_dotenv(), override=True)

        api_key = os.environ["OPENAI_API_KEY"]
        #api_key = st.text_input("OPENAI_API_KEY:", type="password")
        #if api_key:
        #    os.environ["OPENAI_API_KEY"] = api_key
        uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt", "pptx", "jpg", "png"])

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

            data = load_document(file_name)     ### Agregar la conexión load_document
            chunks = chunk_data(data, chunk_size=chunk_size)
            st.write(f"Chunk size: {chunk_size}, Chunks: {len(chunks)}")

            tokens, embedding_cost = calculate_embedding_cost(chunks)
            st.write(f"Embedding cost: ${embedding_cost:.4f}")

            vector_store = create_embeddings(chunks)
            st.session_state.vs = vector_store
            st.session_state.chat_history = []

            st.session_state.vs = vector_store
            st.success("File uploaded, chunked and embedded sucessfully")



    # Agregar el input para la pregunta antes de cargar el archivo
    question = st.text_input("Haz una pregunta acerca del contenido de tu archivo 😀")

    def upload_document_interface():
        document_info = None  # Inicializar como None en caso de que no se cargue ningún archivo
        if uploaded_file:
            with st.spinner("Reading, chunking and embedding file ..."):
                bytes_data = uploaded_file.read()
                document_info = upload_document(bytes_data)
            st.success("File uploaded, chunked and embedded successfully")
        return document_info
    
    # Cambiar el nombre de la función para reflejar mejor su propósito
    document_info = upload_document_interface()
    
    # Luego de cargar el archivo
    if document_info:
        # Cambiar el nombre de la función para reflejar mejor su propósito
        if question:
            
            vector_store = st.session_state.vs
            st.write(f"k: {k}")
            result, st.session_state.chat_history = ask_with_memory(
                vector_store, document_info, st.session_state.chat_history
            )
            answer = answer_question(document_info, question)
            st.text_area("LLM Answer: ", value=answer.answer)

        st.divider()
        if "history" not in st.session_state:
                st.session_state.history = ""
        value = f"Q: {document_info} \nA: {answer}"
        st.session_state.history = (
                f'{value} \n {"-" * 100} \n {st.session_state.history}'
        )
        h = st.session_state.history

if __name__ == "__main__":
    main()
