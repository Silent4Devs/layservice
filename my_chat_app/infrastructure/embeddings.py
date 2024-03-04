import spacy
from typing import List

nlp = spacy.load("en_core_web_sm")

def create_embeddings(chunks: List[str]) -> List[str]:
    embeddings = []
    for chunk in chunks:
        doc = nlp(chunk)
        embeddings.append(doc.vector)
    return embeddings

def chunk_data(data: str, chunk_size: int = 256, chunk_overlap: int = 20) -> List[str]:
    chunks = []
    start = 0
    while start < len(data):
        chunk = data[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks
