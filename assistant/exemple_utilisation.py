"""
Exemple d'utilisation du package PASSE RAG

Ce fichier montre différentes façons d'utiliser le système RAG modulaire.
"""

from Assistant import PasseRAG, create_default_rag, create_custom_rag
from langchain_core.messages import HumanMessage


def exemple_utilisation_simple():
    """Exemple d'utilisation simple avec les paramètres par défaut"""
    print("=== Exemple 1: Utilisation Simple ===")

    # Créer une instance RAG avec les paramètres par défaut
    rag = create_default_rag()

    # Poser une question simple
    question = "Je n'arrive pas à me connecter à mon compte"
    result = rag.query(question)

    print(f"Question: {question}")
    print(f"Réponse: {result['response']}")
    print(f"Classification: {result['classification']}")
    print(f"Succès: {result['success']}")
    print("-" * 50)


def exemple_utilisation_avec_image():
    """Exemple d'utilisation avec une image"""
    print("=== Exemple 2: Utilisation avec Image ===")

    # Créer une instance RAG
    rag = create_default_rag()

    # Poser une question avec une image
    question = "Qu'est-ce qui ne va pas dans cette capture d'écran ?"
    image_path = "faq.jpg"  # Remplacez par le chemin vers votre image

    result = rag.query(question, image_path=image_path)

    print(f"Question: {question}")
    print(f"Image: {image_path}")
    print(f"Réponse: {result['response']}")
    print(f"Classification: {result['classification']}")
    print("-" * 50)


def exemple_utilisation_personnalisee():
    """Exemple d'utilisation avec des paramètres personnalisés"""
    print("=== Exemple 3: Configuration Personnalisée ===")

    # Créer une instance RAG avec des paramètres personnalisés
    rag = create_custom_rag(
        model_name="mistral-small3.1", collection_name="support_docs", k_documents=3
    )

    # Afficher les informations de configuration
    info = rag.get_info()
    print("Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Poser une question
    question = "Comment réinitialiser mon mot de passe ?"
    result = rag.query(question)

    print(f"\nQuestion: {question}")
    print(f"Réponse: {result['response']}")
    print("-" * 50)


def exemple_conversation_avancee():
    """Exemple de conversation avec des messages pré-construits"""
    print("=== Exemple 4: Conversation Avancée ===")

    rag = create_default_rag()

    # Créer des messages pour une conversation
    messages = [
        HumanMessage(content="Bonjour, j'ai un problème avec mon application"),
        # Vous pouvez ajouter d'autres messages ici pour simuler une conversation
    ]

    result = rag.chat(messages)

    print("Messages de conversation:")
    for i, msg in enumerate(messages):
        print(f"  {i + 1}. {msg.content}")

    print(f"\nRéponse: {result['response']}")
    print(f"Classification: {result['classification']}")
    print("-" * 50)


def exemple_utilisation_comme_classe():
    """Exemple d'utilisation directe de la classe PasseRAG"""
    print("=== Exemple 5: Utilisation Directe de la Classe ===")

    # Créer une instance avec tous les paramètres
    rag = PasseRAG(
        model_name="mistral-small3.1",
        embedding_model="nomic-embed-text",
        persist_directory="chroma_db",
        collection_name="faq",
        temperature=0.1,
        k_documents=4,
    )

    # Afficher la configuration
    print("Configuration complète:")
    info = rag.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Utiliser le système
    question = "Quelles sont les heures d'ouverture du support ?"
    result = rag.query(question)

    print(f"\nQuestion: {question}")
    print(f"Réponse: {result['response']}")
    print("-" * 50)


if __name__ == "__main__":
    print("🤖 Exemples d'utilisation du Package PASSE RAG")
    print("=" * 60)

    try:
        # Exécuter les exemples
        exemple_utilisation_simple()
        # exemple_utilisation_avec_image()  # Décommentez si vous avez une image
        exemple_utilisation_personnalisee()
        exemple_conversation_avancee()
        exemple_utilisation_comme_classe()

        print("\n✅ Tous les exemples ont été exécutés avec succès!")

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution des exemples: {e}")
        import traceback

        traceback.print_exc()


# Fonction d'aide pour intégration rapide
def integration_rapide():
    """
    Fonction d'aide pour une intégration rapide dans d'autres projets.

    Returns:
        PasseRAG: Instance prête à utiliser
    """
    return create_default_rag()
