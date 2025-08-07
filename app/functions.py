from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def conn_db(db_name, db_user, db_password, db_host, db_port):

    # Configuration de la base PostgreSQL
    DB_NAME = db_name #"faqdb"
    DB_USER = db_user #"faquser"
    DB_PASSWORD = db_password #"faqpass"
    DB_HOST = db_host #"localhost"
    DB_PORT = db_port #"5432"

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
            
    return get_db
        