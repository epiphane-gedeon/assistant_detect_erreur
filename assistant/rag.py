"""
PASSE RAG Package - A modular RAG system for intelligent assistance

This package provides a reusable RAG (Retrieval-Augmented Generation) system
that can be easily integrated into other applications.
"""

import os
import json
import re
import base64
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from operator import add as add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from assistant.functions import connect_vectorstore


class AgentState(TypedDict):
    """State definition for the RAG agent"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    error_classification: Optional[dict]


class Assistant:
    """
    A modular RAG system for intelligent assistance.

    This class encapsulates all RAG functionality and can be easily instantiated
    and used in different contexts.
    """

    def __init__(
        self,
        model_name: str = "mistral-small3.1",
        embedding_model: str = "nomic-embed-text",
        persist_directory: str = "chroma_db",
        collection_name: str = "faq",
        temperature: float = 0,
        k_documents: int = 5,
        base_url: str="http://localhost:11434",
    ):
        """
        Initialize the PASSE RAG system.

        Args:
            model_name: Name of the LLM model to use
            embedding_model: Name of the embedding model
            persist_directory: Directory for vector store persistence
            collection_name: Name of the vector store collection
            temperature: Temperature setting for the LLM
            k_documents: Number of documents to retrieve
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.temperature = temperature
        self.k_documents = k_documents
        self.base_url = base_url
        self.conversation_state = {
            "messages": [],
            "error_classification": None
        }

        # Initialize components
        self._setup_models()
        self._setup_vectorstore()
        self._setup_tools()
        self._setup_agent()

    def _setup_models(self):
        """Initialize LLM and embedding models"""
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url,
            client_kwargs={
                "headers": {
                    "Authorization": "Bearer fbda85cca422db59caccfff5daae9286f4803bda6440b360ec30e52a8a1a1f77",
                    "Content-Type": "application/json",
                }
            }
        )
        # http://144.6.107.170:21032/?token=b05ddaf8043bff8974f2975b87fc9212c37056e7b138d7dfd6568ce8b7c7b3b2
        # self.llm = ChatOllama(
        #     model=self.model_name,
        #     temperature=self.temperature,
        #     base_url=self.base_url
        # )
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)

    def _setup_vectorstore(self):
        """Initialize vector store and retriever"""
        self.vectorstore = connect_vectorstore(
            self.persist_directory, self.collection_name
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k_documents}
        )

    def _setup_tools(self):
        """Setup tools for the RAG system"""

        @tool
        def retriever_tool(query: str) -> str:
            """
            This tool searches and returns relevant information from the knowledge base.
            """
            docs = self.retriever.invoke(query)

            if not docs:
                return "I found no relevant information in the knowledge base."

            results = []
            for i, doc in enumerate(docs):
                results.append(f"Document {i + 1}:\n{doc.page_content}")

            return "\n\n".join(results)

        @tool
        def classifier_tool(query: str) -> str:
            """
            Classify the error with severity and type.
            """
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content="Tu es un classificateur intelligent. Donne uniquement un JSON avec les clés 'type' et 'severity'."
                    ),
                    HumanMessage(content=query),
                ]
            )

            print(f"[CLASSIFIER TOOL] Classification Response: {response.content}")
            return response.content

        self.tools = [retriever_tool, classifier_tool]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _setup_agent(self):
        """Setup the LangGraph agent"""

        def should_continue(state: AgentState) -> bool:
            """Check if the last message contains tool calls."""
            result = state["messages"][-1]
            return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

        def call_llm(state: AgentState) -> AgentState:
            """Function to call the LLM."""
            messages = list(state["messages"])

            # Detect if there's an image in the conversation
            has_image = self._has_image_content(messages)

            # Choose appropriate system prompt
            system_prompt = self._get_system_prompt(state, has_image)

            # Prepare messages for the model
            useful_messages = [
                msg for msg in messages if isinstance(msg, (HumanMessage, ToolMessage))
            ]

            final_messages = [SystemMessage(content=system_prompt)] + useful_messages

            # Call the model
            message = self.llm_with_tools.invoke(final_messages)
            
            print(f"[CALL LLM] final state: {state}")
            return {
                "messages": [message],
                "error_classification": state.get("error_classification"),
            }

        def take_action(state: AgentState) -> AgentState:
            """Execute tool calls from the LLM's response."""
            tool_calls = state["messages"][-1].tool_calls
            results = []

            for t in tool_calls:
                tool_name = t["name"]
                args = t["args"].get("query", "")

                print(f"\n[TAKE ACTION] Calling Tool: '{tool_name}' with query: {args}")

                if tool_name not in self.tools_dict:
                    result = f"Tool '{tool_name}' not implemented."
                elif tool_name == "classifier_tool":
                    result = self._handle_classification(args, state)
                else:
                    try:
                        result = self.tools_dict[tool_name].invoke(args)
                    except Exception as e:
                        print(f"[TAKE ACTION] Error calling tool '{tool_name}': {e}")
                        result = f"Error in tool {tool_name}"

                results.append(
                    ToolMessage(
                        tool_call_id=t["id"], name=tool_name, content=str(result)
                    )
                )
                print(f"[TAKE ACTION] Result from '{tool_name}': {result}")

            return {
                "messages": results,
                "error_classification": state.get("error_classification"),
            }

        # Create the graph
        graph = StateGraph(AgentState)
        graph.add_node("llm", call_llm)
        graph.add_node("tools", take_action)

        graph.add_conditional_edges("llm", should_continue, {True: "tools", False: END})
        graph.add_edge("tools", "llm")
        graph.set_entry_point("llm")

        self.agent = graph.compile()

    def _has_image_content(self, messages: List[BaseMessage]) -> bool:
        """Check if any message contains image content"""
        for msg in messages:
            if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
                for content_item in msg.content:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "image_url"
                    ):
                        return True
        return False

    def _get_system_prompt(self, state: AgentState, has_image: bool) -> str:
        """Get the appropriate system prompt based on context"""

        standard_prompt = """
        You are PASSE Bot, an intelligent AI assistant who helps solve user problems on the platform based on the data loaded into your knowledge base.
        Use the retriever tool available to answer questions about problems if you can. You can make multiple calls if needed.
        Before answering the question you have to classify an error or problem, use the classifier_tool. But only use it once per conversation.
        If you need to look up some information before asking a follow up question, you are allowed to do that!
        Respond in the language used to ask you the question.
        """

        visual_prompt = """
        You are PASSE Bot, an intelligent AI assistant who helps solve user problems on the platform.

        IMPORTANT: When you receive an image along with text, use the visual information to better understand the context and provide more accurate assistance.

        WORKFLOW:
        1. **Analyze the complete context**: If an image is provided, examine it carefully to understand what the user is showing you (error messages, screenshots, interface elements, etc.). Use this visual context to enrich your understanding of the text question.

        2. **Classify the problem**: Based on both the text AND visual context (if image provided), use the classifier_tool to determine the type and severity of the issue. The visual context should help you classify more accurately.

        3. **Search for solutions**: Use the retriever_tool to find relevant information from your knowledge base. When searching, don't modify the user's question to create the query. Just combine the text and visual context to form a comprehensive query. This will help you find more precise solutions.

        4. **Provide comprehensive answers**: Give solutions that address what you see in the image and what the user describes in text.

        TOOLS AVAILABLE:
        - classifier_tool: Classify problems with type and severity (use context from image + text)
        - retriever_tool: Search knowledge base for solutions

        KEY PRINCIPLES:
        - Use visual context to enhance understanding, not just describe images
        - Integrate image information into your classification and search queries
        - Provide more precise help based on complete context
        - Respond in the language used by the user

        Remember: The image is additional context to help you understand the problem better, not something to simply describe.
        """

        if state.get("error_classification"):
            classification_info = state["error_classification"]
            return f"""
            You are PASSE Bot. The error has been classified as:
            - Type: {classification_info["type"]}
            - Severity: {classification_info["severity"]}

            Now use the retriever tool with user's query to find solutions, citing document sources.
            """
        elif has_image:
            return visual_prompt
        else:
            return standard_prompt

    def _handle_classification(self, args: str, state: AgentState) -> Dict[str, Any]:
        """Handle classification tool results"""
        raw = self.tools_dict["classifier_tool"].invoke(args)
        clean_json = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        try:
            parsed = json.loads(clean_json)
            state["error_classification"] = parsed
            return parsed
        except json.JSONDecodeError:
            print("[CLASSIFICATION] Parsing error in classifier_tool output")
            fallback = {"type": "unknown", "severity": "unknown"}
            state["error_classification"] = fallback
            return fallback

    def query(self, message: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a text message and optionally an image.

        Args:
            message: Text message/question to ask
            image_path: Optional path to an image file

        Returns:
            Dictionary containing the response and metadata
        """
        # Prepare the message content
        content = [{"type": "text", "text": message}]

        # Add image if provided
        if image_path and os.path.exists(image_path):
            base64_image = self._encode_image(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        # Create the human message
        human_message = HumanMessage(content=content)

        # Invoke the agent
        try:
            result = self.agent.invoke({"messages": [human_message]})

            return {
                "response": result["messages"][-1].content,
                "classification": result.get("error_classification"),
                "message_count": len(result["messages"]),
                "success": True,
            }
        except Exception as e:
            return {
                "response": f"Error processing query: {str(e)}",
                "classification": None,
                "message_count": 0,
                "success": False,
                "error": str(e),
            }

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def chat(self, messages) -> Dict[str, Any]:
        """
        Have a conversation with pre-constructed messages.

        Args:
            messages: List of BaseMessage objects

        Returns:
            Dictionary containing the response and metadata
        """
        try:
            print(messages)
            result = self.agent.invoke({"messages": messages})
            
            
            # result_messages = [
            #     msg for msg in result["messages"] if isinstance(msg, (HumanMessage, AIMessage)) and msg.content not in [None, "", " "]
            # ]

            return {
                "response": result["messages"][-1].content,
                # "response": result_messages,
                "classification": result.get("error_classification"),
                "message_count": len(result["messages"]),
                "success": True,
            }
        except Exception as e:
            return {
                "response": f"Error processing chat: {str(e)}",
                "classification": None,
                "message_count": 0,
                "success": False,
                "error": str(e),
            }

    def chat_with_memory(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Chat method that maintains conversation memory between calls.
        """
        # Préparer le contenu
        # content = [{"type": "text", "text": message}]
        
        # if image_path and os.path.exists(image_path):
        #     base64_image = self._encode_image(image_path)
        #     content.append({
        #         "type": "image_url",
        #         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        #     })
        
        # ✅ Ajouter le nouveau message à l'historique existant
        new_message = messages
        self.conversation_state["messages"].append(new_message)
        
        try:
            # ✅ Invoquer avec TOUT l'historique
            result = self.agent.invoke(self.conversation_state)
            
            # ✅ Mettre à jour l'état pour le prochain appel
            self.conversation_state["messages"] = result["messages"]
            self.conversation_state["error_classification"] = result.get("error_classification")
            
            return {
                "response": result["messages"][-1].content,
                "classification": result.get("error_classification"),
                "message_count": len(result["messages"]),
                "conversation_length": len(self.conversation_state["messages"]),
                "success": True,
            }
        except Exception as e:
            return {
                "response": f"Error processing chat: {str(e)}",
                "classification": None,
                "message_count": 0,
                "success": False,
                "error": str(e),
            }

    def clear_conversation(self):
        """Clear the conversation memory"""
        self.conversation_state = {
            "messages": [],
            "error_classification": None
        }
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history"""
        return self.conversation_state["messages"]

    def get_info(self) -> Dict[str, Any]:
        """Get information about the RAG system configuration"""
        return {
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
            "temperature": self.temperature,
            "k_documents": self.k_documents,
            "tools_available": [tool.name for tool in self.tools],
        }


    # A vérifier
    
    def update_faq_documents(self, host : str, port : int, dbname : str, user : str, password : str, table_name: str = "faq") -> Dict[str, Any]:
        """
        Update the vector store with new FAQ documents from database, avoiding duplicates.

        Args:
            table_name: Name of the database table containing FAQ

        Returns:
            Dictionary with update results
        """
        from assistant.functions import update_faq_vectorstore

        return update_faq_vectorstore(
            host=host, 
            port=port, 
            dbname=dbname, 
            user=user, 
            password=password,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            table_name=table_name,
        )

    def get_vectorstore_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current vector store.

        Returns:
            Dictionary with vector store statistics
        """
        from assistant.functions import get_vectorstore_stats

        return get_vectorstore_stats(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

    def add_custom_documents(
        self, documents: List, check_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Add custom documents to the vector store with duplicate checking.

        Args:
            documents: List of Document objects to add
            check_duplicates: Whether to check for and skip duplicates

        Returns:
            Dictionary with operation results
        """
        from assistant.functions import (
            add_documents_to_vectorstore,
            get_vectorstore_stats,
        )

        try:
            # Get stats before adding
            stats_before = get_vectorstore_stats(
                self.persist_directory, self.collection_name
            )
            docs_before = stats_before.get("total_documents", 0)

            # Add documents
            add_documents_to_vectorstore(
                documents=documents,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                check_duplicates=check_duplicates,
            )

            # Get stats after adding
            stats_after = get_vectorstore_stats(
                self.persist_directory, self.collection_name
            )
            docs_after = stats_after.get("total_documents", 0)

            return {
                "success": True,
                "documents_added": docs_after - docs_before,
                "total_documents": docs_after,
                "duplicates_checked": check_duplicates,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "documents_added": 0,
                "total_documents": 0,
            }


# Factory function for easy instantiation
# def create_passe_rag(**kwargs) -> RAG:
#     """
#     Factory function to create a RAG instance with custom parameters.

#     Returns:
#         Configured RAG instance
#     """
#     return RAG(**kwargs)


# # Pre-configured instances for common use cases
# def create_default_rag() -> RAG:
#     """Create a RAG instance with default settings"""
#     return RAG()


# def create_custom_rag(
#     model_name: str, collection_name: str, k_documents: int = 5
# ) -> RAG:
#     """Create a RAG instance with custom model and collection"""
#     return RAG(
#         model_name=model_name, collection_name=collection_name, k_documents=k_documents
#     )
