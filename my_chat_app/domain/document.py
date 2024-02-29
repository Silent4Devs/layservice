from typing import Union, List, Optional  # Agrega Optional aquí
from pydantic import BaseModel

class DocumentInfo(BaseModel):
    notas: Union[str, List[str]]
    extension: str
    content: Optional[str] = None 
