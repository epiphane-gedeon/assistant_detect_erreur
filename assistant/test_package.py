"""
Script de test pour le package PASSE RAG

Ce script teste rapidement si le package fonctionne correctement.
"""

import sys
import os


def test_imports():
    """Test des imports du package"""
    print("🔍 Test des imports...")

    try:
        from Assistant import PasseRAG, create_default_rag, create_custom_rag

        print("✅ Imports réussis")
        return True
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False


def test_instantiation():
    """Test de l'instanciation de la classe"""
    print("🔍 Test de l'instanciation...")

    try:
        from Assistant import create_default_rag

        # Test avec les paramètres par défaut
        rag = create_default_rag()

        # Vérifier que l'objet est créé correctement
        info = rag.get_info()
        print(f"✅ RAG créé avec succès - Modèle: {info['model_name']}")
        return True, rag
    except Exception as e:
        print(f"❌ Erreur d'instanciation: {e}")
        return False, None


def test_configuration():
    """Test de la configuration personnalisée"""
    print("🔍 Test de la configuration personnalisée...")

    try:
        from Assistant import PasseRAG

        rag = PasseRAG(
            model_name="mistral-small3.1",
            collection_name="test_collection",
            temperature=0.1,
            k_documents=3,
        )

        info = rag.get_info()
        print(f"✅ Configuration personnalisée réussie")
        print(f"   - Modèle: {info['model_name']}")
        print(f"   - Collection: {info['collection_name']}")
        print(f"   - Temperature: {info['temperature']}")
        print(f"   - K documents: {info['k_documents']}")
        return True
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return False


def test_basic_query():
    """Test d'une requête basique"""
    print("🔍 Test d'une requête basique...")

    try:
        from Assistant import create_default_rag

        rag = create_default_rag()

        # Question simple pour tester
        question = "Bonjour, pouvez-vous m'aider ?"
        print(f"   Question: {question}")

        result = rag.query(question)

        if result["success"]:
            print("✅ Requête réussie")
            print(f"   Réponse reçue (longueur: {len(result['response'])} caractères)")
            print(
                f"   Classification: {result.get('classification', 'Non disponible')}"
            )
            return True
        else:
            print(f"❌ Requête échouée: {result.get('error', 'Erreur inconnue')}")
            return False

    except Exception as e:
        print(f"❌ Erreur lors de la requête: {e}")
        return False


def test_factory_functions():
    """Test des fonctions factory"""
    print("🔍 Test des fonctions factory...")

    try:
        from Assistant import create_default_rag, create_custom_rag, create_passe_rag

        # Test create_default_rag
        rag1 = create_default_rag()
        print("✅ create_default_rag() fonctionne")

        # Test create_custom_rag
        rag2 = create_custom_rag(
            model_name="mistral-small3.1", collection_name="test_custom"
        )
        print("✅ create_custom_rag() fonctionne")

        # Test create_passe_rag
        rag3 = create_passe_rag(temperature=0.2)
        print("✅ create_passe_rag() fonctionne")

        return True
    except Exception as e:
        print(f"❌ Erreur avec les fonctions factory: {e}")
        return False


def run_all_tests():
    """Exécute tous les tests"""
    print("🤖 PASSE RAG - Tests du Package")
    print("=" * 50)

    tests_results = []

    # Test 1: Imports
    tests_results.append(test_imports())
    print()

    # Test 2: Instanciation
    success, rag = test_instantiation()
    tests_results.append(success)
    print()

    # Test 3: Configuration
    tests_results.append(test_configuration())
    print()

    # Test 4: Fonctions factory
    tests_results.append(test_factory_functions())
    print()

    # Test 5: Requête basique (seulement si l'instanciation a réussi)
    if success and rag:
        tests_results.append(test_basic_query())
        print()

    # Résumé
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)

    passed = sum(tests_results)
    total = len(tests_results)

    print(f"Tests réussis: {passed}/{total}")

    if passed == total:
        print("🎉 Tous les tests sont passés ! Le package est prêt à être utilisé.")
        return True
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez la configuration.")
        return False


if __name__ == "__main__":
    # Ajouter le répertoire parent au path pour les imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    success = run_all_tests()

    if success:
        print("\n🚀 Le package PASSE RAG est prêt !")
        print("\nPour commencer à l'utiliser :")
        print("```python")
        print("from Assistant import create_default_rag")
        print("rag = create_default_rag()")
        print("result = rag.query('Votre question')")
        print("print(result['response'])")
        print("```")
    else:
        print("\n❌ Le package nécessite des corrections avant utilisation.")
        sys.exit(1)
