# application/use_cases.py

from domain.document import DocumentInfo
from domain.chat_interaction import ChatQuestion, ChatAnswer
from infrastructure.document_loader import load_document
from infrastructure.embeddings import create_embeddings, chunk_data
from infrastructure.chat_interface import ask_and_get_answer


def upload_document(file_path: str) -> DocumentInfo:
    return load_document(file_path)

def answer_question(document_info: DocumentInfo, question: str) -> ChatAnswer:
    chunks = chunk_data(document_info.content)
    embeddings = create_embeddings(chunks)
    answer = ask_and_get_answer(embeddings, question)
    return ChatAnswer(answer)
