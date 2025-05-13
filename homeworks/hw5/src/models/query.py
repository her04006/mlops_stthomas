from pydantic import BaseModel
from typing import List

class RAGRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answers: List[str]
