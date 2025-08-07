"""
Exemple d'utilisation du package PASSE RAG depuis l'extérieur

Ce fichier démontre comment utiliser le package depuis un autre projet.
"""

# Import du package PASSE RAG
from Assistant import create_default_rag, create_custom_rag, PasseRAG


def exemple_integration_simple():
    """Exemple d'intégration simple dans une application"""
    print("🚀 Démarrage de l'application avec PASSE RAG")

    # Créer une instance RAG
    rag = create_default_rag()

    # Afficher les informations du système
    info = rag.get_info()
    print("Configuration RAG:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)

    # Simuler une session de chat
    questions = [
        "Bonjour ! Je peux vous aider ?",
        "Je n'arrive pas à me connecter à mon compte",
        "Mon application plante au démarrage",
        "Comment réinitialiser mon mot de passe ?",
        "Merci pour votre aide !",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n👤 Utilisateur {i}: {question}")

        # Interroger le RAG
        result = rag.query(question)

        if result["success"]:
            print(f"🤖 PASSE Bot: {result['response'][:200]}...")

            # Afficher la classification si disponible
            if result["classification"]:
                print(f"📊 Classification: {result['classification']}")
        else:
            print(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")

        print("-" * 30)


def exemple_api_flask():
    """Exemple d'intégration dans une API Flask"""
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    # Initialiser le RAG au démarrage de l'application
    print("🔧 Initialisation du RAG pour l'API...")
    rag = create_default_rag()
    print("✅ RAG initialisé avec succès")

    @app.route("/health", methods=["GET"])
    def health_check():
        """Point de santé de l'API"""
        info = rag.get_info()
        return jsonify({"status": "healthy", "rag_config": info})

    @app.route("/ask", methods=["POST"])
    def ask_question():
        """Endpoint pour poser des questions"""
        try:
            data = request.json
            question = data.get("question", "")

            if not question:
                return jsonify({"error": "Question is required"}), 400

            # Interroger le RAG
            result = rag.query(question)

            if result["success"]:
                return jsonify(
                    {
                        "answer": result["response"],
                        "classification": result["classification"],
                        "success": True,
                    }
                )
            else:
                return jsonify(
                    {"error": result.get("error", "Unknown error"), "success": False}
                ), 500

        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 500

    @app.route("/ask-with-image", methods=["POST"])
    def ask_with_image():
        """Endpoint pour poser des questions avec une image"""
        try:
            data = request.json
            question = data.get("question", "")
            image_path = data.get("image_path", "")

            if not question:
                return jsonify({"error": "Question is required"}), 400

            # Interroger le RAG avec image si fournie
            if image_path:
                result = rag.query(question, image_path=image_path)
            else:
                result = rag.query(question)

            if result["success"]:
                return jsonify(
                    {
                        "answer": result["response"],
                        "classification": result["classification"],
                        "has_image": bool(image_path),
                        "success": True,
                    }
                )
            else:
                return jsonify(
                    {"error": result.get("error", "Unknown error"), "success": False}
                ), 500

        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 500

    return app


def exemple_classe_service():
    """Exemple d'encapsulation dans une classe de service"""

    class SupportService:
        """Service d'assistance utilisant PASSE RAG"""

        def __init__(self, custom_config=None):
            """Initialiser le service avec une configuration optionnelle"""
            if custom_config:
                self.rag = PasseRAG(**custom_config)
            else:
                self.rag = create_default_rag()

            self.session_history = []

        def get_help(self, question, user_id=None, include_history=False):
            """Obtenir de l'aide pour une question"""

            # Ajouter à l'historique
            if user_id:
                self.session_history.append(
                    {
                        "user_id": user_id,
                        "question": question,
                        "timestamp": __import__("datetime").datetime.now(),
                    }
                )

            # Interroger le RAG
            result = self.rag.query(question)

            # Préparer la réponse
            response = {
                "answer": result["response"],
                "classification": result["classification"],
                "success": result["success"],
                "user_id": user_id,
            }

            if include_history:
                response["session_history"] = self.session_history[
                    -5:
                ]  # Dernières 5 questions

            return response

        def get_help_with_image(self, question, image_path, user_id=None):
            """Obtenir de l'aide avec une image"""
            result = self.rag.query(question, image_path=image_path)

            return {
                "answer": result["response"],
                "classification": result["classification"],
                "success": result["success"],
                "user_id": user_id,
                "has_image": True,
            }

        def get_stats(self):
            """Obtenir les statistiques du service"""
            return {
                "total_questions": len(self.session_history),
                "rag_config": self.rag.get_info(),
                "unique_users": len(
                    set(
                        item.get("user_id")
                        for item in self.session_history
                        if item.get("user_id")
                    )
                ),
            }

    # Utilisation du service
    print("🛠️  Création du service d'assistance...")

    # Configuration personnalisée
    custom_config = {
        "temperature": 0.1,
        "k_documents": 3,
        "collection_name": "support_advanced",
    }

    service = SupportService(custom_config)

    # Test du service
    questions_test = [
        ("user123", "Comment installer l'application ?"),
        ("user456", "Je reçois une erreur 404"),
        ("user123", "L'installation a échoué, que faire ?"),
    ]

    print("🧪 Test du service...")
    for user_id, question in questions_test:
        response = service.get_help(question, user_id=user_id, include_history=True)
        print(f"👤 {user_id}: {question}")
        print(f"🤖 Réponse: {response['answer'][:100]}...")
        if response["classification"]:
            print(f"📊 Classification: {response['classification']}")
        print("-" * 40)

    # Afficher les statistiques
    stats = service.get_stats()
    print("\n📈 Statistiques du service:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    """Fonction principale pour exécuter les exemples"""
    print("🤖 Exemples d'intégration du package PASSE RAG")
    print("=" * 60)

    try:
        # Exemple 1: Intégration simple
        print("\n1️⃣  EXEMPLE: Intégration Simple")
        print("-" * 40)
        exemple_integration_simple()

        # Exemple 2: Service encapsulé
        print("\n\n2️⃣  EXEMPLE: Classe de Service")
        print("-" * 40)
        exemple_classe_service()

        # Exemple 3: Information sur l'API Flask
        print("\n\n3️⃣  EXEMPLE: API Flask")
        print("-" * 40)
        print("Code d'exemple pour une API Flask créé.")
        print("Pour l'utiliser, exécutez:")
        print("```python")
        print("app = exemple_api_flask()")
        print("app.run(debug=True)")
        print("```")
        print("Endpoints disponibles:")
        print("  - GET  /health")
        print("  - POST /ask")
        print("  - POST /ask-with-image")

        print("\n✅ Tous les exemples d'intégration sont prêts !")

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution des exemples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
