import streamlit as st
from domain.document_loaders  import load_document
from domain.text_chunker import chunk_data
from domain.embeddings import create_embeddings
from domain.question_answer import ask_and_get_answer
from domain.cost_calculator import calculate_embedding_cost
from domain.memory import clear_history, ask_with_memory
import os

def run():
    st.image("my_chat_app/public/s4b_logo.png")
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
        from dotenv import load_dotenv
        
        load_dotenv()

        api_key = os.environ["OPENAI_API_KEY"]

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

if __name__ == "__main__":
    run()
