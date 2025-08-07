from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base



# Base pour les modèles
Base = declarative_base()


# Modèle FAQ
class FAQ(Base):
    __tablename__ = "faq"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    question = Column(String, nullable=False)
    procede = Column(Text, nullable=False)

