from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Votre frontend Angular
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app import routes