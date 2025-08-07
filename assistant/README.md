# Package PASSE RAG

Un système RAG (Retrieval-Augmented Generation) modulaire et réutilisable pour l'assistance intelligente.

## 🚀 Installation et Configuration

### Prérequis
- Python 3.8+
- Les dépendances listées dans `requirements.txt`
- Ollama installé avec les modèles `mistral-small3.1` et `nomic-embed-text`

### Installation des dépendances
```bash
pip install -r requirements.txt
```

## 📖 Utilisation

### 1. Utilisation Simple

```python
from Assistant import create_default_rag

# Créer une instance RAG
rag = create_default_rag()

# Poser une question
result = rag.query("Je n'arrive pas à me connecter")
print(result['response'])
```

### 2. Utilisation avec Image

```python
from Assistant import create_default_rag

rag = create_default_rag()

# Poser une question avec une image
result = rag.query(
    "Qu'est-ce qui ne va pas dans cette capture ?",
    image_path="screenshot.jpg"
)
print(result['response'])
```

### 3. Configuration Personnalisée

```python
from Assistant import PasseRAG

# Créer une instance avec des paramètres personnalisés
rag = PasseRAG(
    model_name="mistral-small3.1",
    collection_name="ma_base_docs",
    k_documents=3,
    temperature=0.1
)

result = rag.query("Ma question")
```

### 4. Factory Functions

```python
from Assistant import create_custom_rag

# Utiliser les fonctions factory pour plus de simplicité
rag = create_custom_rag(
    model_name="mistral-small3.1",
    collection_name="support_docs"
)
```

## 🔧 API Reference

### Classe `PasseRAG`

#### Constructeur
```python
PasseRAG(
    model_name: str = "mistral-small3.1",
    embedding_model: str = "nomic-embed-text", 
    persist_directory: str = "chroma_db",
    collection_name: str = "faq",
    temperature: float = 0,
    k_documents: int = 5
)
```

#### Méthodes Principales

##### `query(message: str, image_path: Optional[str] = None) -> Dict`
Pose une question au système RAG.

**Paramètres:**
- `message`: Question ou message texte
- `image_path`: Chemin optionnel vers une image

**Retourne:**
```python
{
    "response": "Réponse du système",
    "classification": {"type": "...", "severity": "..."},
    "message_count": 3,
    "success": True
}
```

##### `chat(messages: List[BaseMessage]) -> Dict`
Conversation avec des messages pré-construits.

##### `get_info() -> Dict`
Retourne les informations de configuration du système.

### Fonctions Factory

##### `create_default_rag() -> PasseRAG`
Crée une instance avec les paramètres par défaut.

##### `create_custom_rag(model_name: str, collection_name: str, k_documents: int = 5) -> PasseRAG`
Crée une instance avec des paramètres personnalisés.

##### `create_passe_rag(**kwargs) -> PasseRAG`
Fonction factory générique acceptant tous les paramètres.

## 🎯 Exemples d'Intégration

### Dans une API Flask

```python
from flask import Flask, request, jsonify
from Assistant import create_default_rag

app = Flask(__name__)
rag = create_default_rag()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    
    result = rag.query(question)
    
    return jsonify({
        'answer': result['response'],
        'classification': result['classification']
    })
```

### Dans une Application Streamlit

```python
import streamlit as st
from Assistant import create_default_rag

@st.cache_resource
def load_rag():
    return create_default_rag()

rag = load_rag()

# Interface utilisateur
question = st.text_input("Posez votre question:")
uploaded_file = st.file_uploader("Optionnel: Joindre une image")

if st.button("Envoyer"):
    if uploaded_file:
        # Sauvegarder l'image temporairement
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        result = rag.query(question, "temp_image.jpg")
    else:
        result = rag.query(question)
    
    st.write(result['response'])
```

### Dans un Bot Discord

```python
import discord
from Assistant import create_default_rag

class RAGBot(discord.Client):
    def __init__(self):
        super().__init__()
        self.rag = create_default_rag()
    
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        if message.content.startswith('!ask'):
            question = message.content[5:]
            result = self.rag.query(question)
            await message.channel.send(result['response'])
```

## 🛠️ Fonctionnalités

### ✅ Implémentées
- Classification automatique des erreurs
- Recherche sémantique dans la base de connaissances
- Support des images (analyse visuelle)
- Interface modulaire et réutilisable
- Configuration flexible
- Gestion d'état avec LangGraph
- Outils de classification et de récupération

### 🔮 Améliorations Futures
- Cache intelligent des réponses
- Support multi-langue automatique
- Métriques et monitoring
- Interface de configuration web
- Support de multiples bases vectorielles
- Streaming des réponses

## 📁 Structure des Fichiers

```
Assistant/
├── __init__.py          # Exports et configuration du package
├── passe_rag.py         # Classe principale PasseRAG
├── exemple_utilisation.py # Exemples d'utilisation
├── functions.py         # Fonctions utilitaires
├── rag.py              # Ancienne implémentation (référence)
└── README.md           # Cette documentation
```

## 🐛 Dépannage

### Erreurs Communes

1. **"Model not found"**
   - Vérifiez que les modèles Ollama sont installés
   - `ollama pull mistral-small3.1`
   - `ollama pull nomic-embed-text`

2. **"Vector store not found"**
   - Assurez-vous que la base vectorielle existe
   - Vérifiez le chemin `persist_directory`

3. **Erreurs d'import**
   - Vérifiez que toutes les dépendances sont installées
   - `pip install -r requirements.txt`

### Logs et Debug

Le système affiche des logs détaillés pendant l'exécution. Pour plus de verbosité :

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contribution

Pour contribuer au package :

1. Créez une branche pour votre fonctionnalité
2. Ajoutez des tests si nécessaire
3. Documentez vos changements
4. Soumettez une pull request

## 📄 Licence

Ce projet est sous licence [votre licence ici].
