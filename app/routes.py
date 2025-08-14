from app import app
from app.functions import conn_db
from app.models import FAQ
from app.validators import FAQResponse
from assistant import Assistant 
from typing import Union, List, Optional
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException
from langchain_core.messages import HumanMessage

get_db = conn_db("faqdb", "faquser", "faqpass", "localhost", "5432")

db_session = get_db

assist=Assistant(base_url="https://utc-advertisement-expensive-provided.trycloudflare.com")



from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    messages = [HumanMessage(
        content=[
            {"type": "text", "text": request.question},
        ]
    )]
    
    # [HumanMessage(
    #     content=[
    #         {"type": "text", "text": "Please provide a detailed answer."},
    #     ]
    # )]
    
    result = assist.chat(messages)
    # print(result)
    return result

# @app.get("/ask")
# def read_root():
#     test_question = "Je n'arrive pas à me connecter à mon compte"

#     messages = [HumanMessage(
#         content=[
#             {"type": "text", "text": test_question},
#             # {
#             #     "type": "image_url",
#             #     "image_url": {
#             #         "url": f"data:image/jpeg;base64,{base64_image}"
#             #     },
#             # },
#         ]
#     )]

#     result = assist.chat(messages)
#     return result