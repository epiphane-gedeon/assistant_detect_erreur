import os
import json
import re
import base64
import psycopg2
import requests
from langchain_core.documents import Document
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from operator import add as add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from requests.auth import HTTPBasicAuth


def conn_db(host, port, dbname, user, password):
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,  # 🔐 Remplace par ta valeur réelle
    )
    return conn


def connect_vectorstore(
    persist_directory: str,
    collection_name: str,
    embeddings=OllamaEmbeddings(model="nomic-embed-text"),
):
    """Connect to the vector store and return the collection."""

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vectorstore


def make_documents(table_name="faq"):
    """_summary_

    Args:
        table_name (str, optional): _description_. Defaults to "faq".

    Returns:
        Document : document
    """
    conn = conn_db("localhost", "5432", "faqdb", "faquser", "faqpass")
    cur = conn.cursor()
    cur.execute("SELECT question, procede FROM " + table_name + ";")

    rows = cur.fetchall()

    # Création des documents LangChain
    documents = [
        Document(
            page_content=row[0] + " : " + row[1],
            metadata={"question": row[0], "source": "faq : " + row[0]},
        )
        for row in rows
    ]

    cur.close()
    conn.close()

    return documents


def encode_image(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def check_document_exists(
    vectorstore: Chroma, document: Document, similarity_threshold: float = 0.95
) -> bool:
    """
    Check if a document already exists in the vector store based on similarity.

    Args:
        vectorstore: The Chroma vectorstore instance
        document: The document to check
        similarity_threshold: Minimum similarity score to consider as duplicate (default: 0.95)

    Returns:
        bool: True if document exists, False otherwise
    """
    try:
        # Search for similar documents
        results = vectorstore.similarity_search_with_score(document.page_content, k=3)

        # Check if any result exceeds the similarity threshold
        for doc, score in results:
            # Convert distance to similarity (Chroma returns distance, lower is more similar)
            similarity = 1 - score
            if similarity >= similarity_threshold:
                # Check if it's the same source or question
                if doc.metadata.get("question") == document.metadata.get(
                    "question"
                ) or doc.metadata.get("source") == document.metadata.get("source"):
                    return True

        return False
    except Exception as e:
        print(f"Error checking document existence: {e}")
        return False


def add_documents_to_vectorstore(
    documents: Sequence[Document],
    persist_directory: str,
    collection_name: str,
    check_duplicates: bool = True,
    similarity_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Add documents to an existing vector store with duplicate detection.

    Args:
        documents: Documents to add
        persist_directory: Directory for vector store persistence
        collection_name: Name of the collection
        check_duplicates: Whether to check for duplicates before adding
        similarity_threshold: Threshold for duplicate detection

    Returns:
        Dictionary with operation results
    """
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    # Connect to the existing vector store
    vectorstore = connect_vectorstore(persist_directory, collection_name)

    added_count = 0
    skipped_count = 0
    unique_sources = set()

    try:
        for doc in documents:
            if check_duplicates and check_document_exists(
                vectorstore, doc, similarity_threshold
            ):
                skipped_count += 1
                # print(
                #     f"⏭️  Document skipped (duplicate): {doc.metadata.get('question', 'Unknown')}"
                # )
            else:
                vectorstore.add_documents([doc])
                added_count += 1
                unique_sources.add(doc.metadata.get("source", "Unknown"))
                print(f"✅ Document added: {doc.metadata.get('question', 'Unknown')}")

        return {
            "success": True,
            "total_documents": len(documents),
            "documents_added": added_count,
            "documents_skipped": skipped_count,
            "unique_sources": len(unique_sources),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_documents": len(documents),
            "documents_added": added_count,
            "documents_skipped": skipped_count,
            "unique_sources": len(unique_sources),
        }


def update_faq_vectorstore(
    host: str = "localhost",
    port: str = "5432",
    dbname: str = "faqdb",
    user: str = "faquser",
    password: str = "faqpass",
    persist_directory: str = "chroma_db",
    collection_name: str = "faq",
    table_name: str = "faq",
    check_duplicates: bool = True,
) -> Dict[str, Any]:
    """
    Update the vector store with latest FAQ data from PostgreSQL.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        dbname: Database name
        user: Database user
        password: Database password
        persist_directory: Vector store directory
        collection_name: Collection name
        table_name: Table name to query
        check_duplicates: Whether to check for duplicates

    Returns:
        Dictionary with update results
    """
    try:
        # Get documents from database
        documents = make_documents(table_name)

        # Add to vector store with duplicate checking
        result = add_documents_to_vectorstore(
            documents=documents,
            persist_directory=persist_directory,
            collection_name=collection_name,
            check_duplicates=check_duplicates,
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_documents": 0,
            "unique_sources": 0,
        }


def get_vectorstore_stats(
    persist_directory: str, collection_name: str
) -> Dict[str, Any]:
    """
    Get statistics about the vector store.

    Args:
        persist_directory: Vector store directory
        collection_name: Collection name

    Returns:
        Dictionary with statistics
    """
    try:
        vectorstore = connect_vectorstore(persist_directory, collection_name)
        collection = vectorstore._collection

        # Get document count
        total_docs = collection.count()

        # Get sample documents to analyze sources
        if total_docs > 0:
            sample_size = min(100, total_docs)  # Sample up to 100 docs
            results = collection.get(limit=sample_size)

            sources = set()
            if results.get("metadatas"):
                for metadata in results["metadatas"]:
                    if metadata and "source" in metadata:
                        sources.add(metadata["source"])

            unique_sources = len(sources)
        else:
            unique_sources = 0

        return {
            "success": True,
            "total_documents": total_docs,
            "unique_sources": unique_sources,
            "collection_name": collection_name,
            "persist_directory": persist_directory,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_documents": 0,
            "unique_sources": 0,
        }


def create_openproject_ticket(
    openproject_url: str,
    api_key: str,
    project_id: int,
    subject: str,
    description: str,
    severity: str = "medium",
    priority_id: Optional[int] = None,
    type_id: Optional[int] = None,
    parent_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Crée un ticket dans OpenProject avec la réponse de l'IA comme description

    Args:
        openproject_url: URL de base d'OpenProject (ex: "https://openproject.example.com")
        api_key: Clé API OpenProject
        project_id: ID du projet OpenProject
        subject: Titre du ticket
        description: Description du ticket (réponse de l'IA)
        severity: Sévérité du ticket ("low", "medium", "high", "critical")
        priority_id: ID de priorité (optionnel)
        type_id: ID du type de ticket (optionnel, par défaut = Task)

    Returns:
        Dict contenant la réponse de l'API OpenProject
    """

    # Mapping des sévérités vers les priorités OpenProject
    severity_to_priority = {
        "low": 1,  # Faible
        "medium": 2,  # Normale
        "high": 3,  # Élevée
        "critical": 4,  # Immédiate
    }

    # Configuration des headers
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Déterminer la priorité basée sur la sévérité
    if priority_id is None:
        priority_id = severity_to_priority.get(
            severity.lower(), 2
        )  # Par défaut: normale

    # Payload pour créer le ticket
    payload = {
        "subject": subject,
        "description": {"format": "markdown", "raw": description},
        "_links": {
            "project": {"href": f"/api/v3/projects/{project_id}"},
            "priority": {"href": f"/api/v3/priorities/{priority_id}"},
        },
        "parent": {"href": f"/api/v3/work_packages/{parent_id}" if parent_id else None},
    }

    # Ajouter le type si spécifié
    if type_id:
        payload["_links"]["type"] = {"href": f"/api/v3/types/{type_id}"}

    try:
        # Envoyer la requête POST
        response = requests.post(
            f"{openproject_url}/api/v3/work_packages",
            headers=headers,
            json=payload,
            timeout=30,
            auth=HTTPBasicAuth("apikey", api_key),
        )

        print(f"🔗 URL de la requête: {response}")

        # Vérifier le statut
        if response.status_code == 201:
            print("Ticket créé avec succès dans OpenProject!")
            ticket_data = response.json()
            print(f"ID du ticket: {ticket_data.get('id')}")
            print(f"Titre: {ticket_data.get('subject')}")
            print(
                f"Priorité: {ticket_data.get('_links', {}).get('priority', {}).get('title', 'Non définie')}"
            )
            return ticket_data
        else:
            print(f"Erreur lors de la création du ticket: {response.status_code}")
            print(f"Détails: {response.text}")
            return {
                "error": True,
                "status_code": response.status_code,
                "message": response.text,
            }

    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion: {e}")
        return {"error": True, "message": f"Erreur de connexion: {str(e)}"}
