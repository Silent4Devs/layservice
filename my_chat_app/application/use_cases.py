from infrastructure.document_loader import load_document_from_upload
from domain.document import DocumentInfo
from domain.document import DocumentInfo
from infrastructure.repositories import Repository

def upload_document_interface(bytes_data: bytes, question: str = "") -> DocumentInfo:
    return load_document_from_upload(bytes_data, question)

def answer_question_interface(question: str, document_info: DocumentInfo, repository: Repository) -> str:
    if question.lower() == "hola":
        return "¡Hola! ¿Cómo estás? ¿En qué puedo ayudarte?"
    elif question.lower() == "como te llamas":
        return "¡LayChat! Mi objetivo es brindarte información!"
    elif question.lower() == "quién es el supervisor":
        return repository.get_supervisor_info()
    else:
        return "Respuesta a la pregunta: " + question
