# domain/chat_interaction.py

from pydantic import BaseModel

class ChatQuestion(BaseModel):
    question: str

class ChatAnswer(BaseModel):
    answer: str
