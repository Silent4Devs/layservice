from pydantic import BaseModel
from typing import Union, List

class DocumentInfo(BaseModel):
    notas: Union[str, List[str]]
    extension: str
    content: str
