# Guide d'utilisation rapide - Package PASSE RAG

## 🎯 Utilisation en 3 étapes

### 1. Import simple
```python
from Assistant import create_default_rag
```

### 2. Création d'instance
```python
rag = create_default_rag()
```

### 3. Utilisation
```python
result = rag.query("Ma question")
print(result['response'])
```

## 📋 Exemples pratiques

### Utilisation basique
```python
from Assistant import create_default_rag

# Créer le RAG
rag = create_default_rag()

# Poser une question
result = rag.query("Je n'arrive pas à me connecter")

# Afficher la réponse
if result['success']:
    print("Réponse:", result['response'])
    print("Classification:", result['classification'])
else:
    print("Erreur:", result['error'])
```

### Avec configuration personnalisée
```python
from Assistant import PasseRAG

# Configuration personnalisée
rag = PasseRAG(
    model_name="mistral-small3.1",
    collection_name="ma_base",
    k_documents=3,
    temperature=0.1
)

result = rag.query("Ma question")
```

### Avec image
```python
from Assistant import create_default_rag

rag = create_default_rag()

result = rag.query(
    "Qu'est-ce qui ne va pas ?",
    image_path="screenshot.jpg"
)
```

### Dans une classe
```python
from Assistant import create_default_rag

class MonAssistant:
    def __init__(self):
        self.rag = create_default_rag()
    
    def aider(self, question):
        return self.rag.query(question)

# Utilisation
assistant = MonAssistant()
reponse = assistant.aider("Comment faire ?")
```

## 🔧 API de retour

Toutes les méthodes `query()` retournent un dictionnaire avec :

```python
{
    "response": "La réponse du bot",
    "classification": {"type": "...", "severity": "..."},
    "message_count": 3,
    "success": True  # ou False en cas d'erreur
}
```

## 🚀 Prêt à utiliser !

Le package est maintenant modulaire et réutilisable. Vous pouvez :

1. **L'importer** dans n'importe quel projet
2. **Le configurer** selon vos besoins
3. **L'intégrer** dans des APIs, interfaces, bots...
4. **Le tester** avec le script de test fourni

### Test rapide
```bash
cd Assistant
python test_package.py
```

### Voir tous les exemples
```bash
python demo_integration.py
```
