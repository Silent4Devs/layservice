import os
from typing import Optional, Union, List
from pydantic import BaseModel

class DocumentInfo(BaseModel):
    notas: Union[str, List[str]]
    extension: str
    content: Optional[str] = None 

def load_document(file):
    import os
    from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredImageLoader
    
    notas, extension = os.path.splitext(file)

    def load_pdf(file):
        print(f"Loading {file}")
        loader = PyPDFLoader(file)
        return loader.load()

    def load_docx(file):
        print(f"Loading {file}")
        loader = Docx2txtLoader(file)
        return loader.load()

    def load_txt(file):
        print(f"Loading {file}")
        loader = TextLoader(file)
        return loader.load()

    def load_pptx(file):
        print(f"Loading {file}")
        loader = UnstructuredPowerPointLoader(file)
        return loader.load()
    
    def load_image(file):
        print(f"Loading {file}")
        loader = UnstructuredImageLoader(file)
        return loader.load()

    # Asigna las funciones de carga a sus respectivas extensiones
    loader_functions = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".pptx": load_pptx,
        ".jpg": load_image,
        ".png": load_image,
    }

    loader_function = loader_functions.get(extension, None)
    if loader_function:
        document_info = loader_function(file)
        return [document_info]  # Devuelve una lista de un solo elemento
    else:
        print(f"Documento '{file}' no soportado o tipo de archivo '{extension}' no reconocido")
        return []  # Devuelve una lista vac

