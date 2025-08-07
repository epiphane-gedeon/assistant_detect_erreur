from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configuration de la base PostgreSQL
DB_NAME = "faqdb"
DB_USER = "faquser"
DB_PASSWORD = "faqpass"
DB_HOST = "localhost"
DB_PORT = "5432"

# Configuration SQLAlchemy
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Création de l'engine SQLAlchemy
engine = create_engine(DATABASE_URL)

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Fonction pour obtenir une session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
