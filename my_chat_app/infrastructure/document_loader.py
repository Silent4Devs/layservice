from domain.document import DocumentInfo

def load_document_from_upload(bytes_data: bytes, question: str) -> DocumentInfo:
    # Lógica para cargar el documento desde los bytes
    content = "Contenido del documento"  # Placeholder
    extension = ".pdf"  # Placeholder
    
    # Lógica para responder a la pregunta personalizada
    if "hola, ¿cómo estás?" in question.lower():
        return "Bien, ¿y tú? ¿En qué puedo ayudarte?"
    else:
        return DocumentInfo(notas=content, extension=extension)
