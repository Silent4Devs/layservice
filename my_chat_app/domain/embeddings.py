from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store
