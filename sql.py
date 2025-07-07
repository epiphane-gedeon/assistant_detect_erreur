import psycopg2

# Configuration de la base PostgreSQL
DB_NAME = "faqdb"
DB_USER = "faquser"
DB_PASSWORD = "faqpass"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_all_faq():
    try:
        # Connexion à la base de données
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )

        # Création d'un curseur
        cur = conn.cursor()

        # Requête SELECT
        cur.execute("SELECT id, question, procede FROM FAQ ORDER BY id;")
        faqs = cur.fetchall()

        # Affichage des résultats
        for faq in faqs:
            print(f"\n❓ Question {faq[0]}:\n{faq[1]}\n➡️ Procédé :\n{faq[2]}")

        # Fermeture
        cur.close()
        conn.close()

    except Exception as e:
        print("Erreur lors de la connexion ou de la requête :", e)

if __name__ == "__main__":
    get_all_faq()
