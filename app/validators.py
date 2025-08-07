from pydantic import BaseModel
from typing import Optional

class FAQResponse(BaseModel):
    id: Optional[int] = None
    question: str
    procede: str

    class Config:
        from_attributes = True