import os
import json
import re
import base64
import psycopg2
from Assistant.functions import conn_db, connect_vectorstore, add_documents_to_vectorstore, make_documets
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

# 1. Initialisation des modèles
llm = ChatOllama(model="mistral-small3.1", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# RAG(llm,vectorstore,embeddings)

persist_directory = "chroma_db"  # Chemin vers le répertoire de persistance
collection_name = "faq"

vectorstore = connect_vectorstore(persist_directory, collection_name)

class RAG:
    def __init__(self, llm: ChatOllama, vectorstore: Chroma, embeddings: OllamaEmbeddings):
        self.llm = llm
        self.vectorstore = vectorstore
        self.embeddings = embeddings


        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} # K is the amount of chunks to return
        )

        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
            error_classification: Optional[dict]    


        @tool
        def retriever_tool(query: str) -> str:
            """
            This tool searches and returns the information from the Stock Market Performance 2024 document.
            """

            docs = retriever.invoke(query)

            if not docs:
                return "I found no relevant information in the Stock Market Performance 2024 document."
            
            results = []
            for i, doc in enumerate(docs):
                results.append(f"Document {i+1}:\n{doc.page_content}")
            
            return "\n\n".join(results)


        @tool
        def classifer_tool(query: str) -> str:
            """
            Classify the error with severity and type.
            """
            response = llm.invoke([
                SystemMessage(content="Tu es un classificateur intelligent. Donne uniquement un JSON avec les clés 'type' et 'severity'."),
                HumanMessage(content=query)
            ])
            
            print(f"[CLASSIFER TOOL] Classification Response: {response.content}")
            
            return response.content


        tools = [retriever_tool,classifer_tool]

        llm = llm.bind_tools(tools)


        def should_continue(state: AgentState):
            """Check if the last message contains tool calls."""
            print("[SHOULD CONTINUE] Checking if we should continue...")
            result = state['messages'][-1]
            return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

        system_prompt = """
        You are PASSE Bot, an intelligent AI assistant who helps on problems of users on the platform based on the data loaded into your knowledge base.
        Use the retriever tool available to answer questions about the every problem if you can. You can make multiple calls if needed.
        Before answering the question you have to classify an error or problem, use the classifer tool. But only use it once per conversation.
        If you need to look up some information before asking a follow up question, you are allowed to do that!
        Please always cite the specific parts of the documents you use in your answers. And respond in the language used to ask you the question.
        If you receive an image, always give a description of it.
        """

        system_prompt_with_visual_context = """
        You are PASSE Bot, an intelligent AI assistant who helps solve user problems on the platform.

        IMPORTANT: When you receive an image along with text, use the visual information to better understand the context and provide more accurate assistance.

        WORKFLOW:
        1. **Analyze the complete context**: If an image is provided, examine it carefully to understand what the user is showing you (error messages, screenshots, interface elements, etc.). Use this visual context to enrich your understanding of the text question.

        2. **Classify the problem**: Based on both the text AND visual context (if image provided), use the classifer_tool to determine the type and severity of the issue. The visual context should help you classify more accurately.

        3. **Search for solutions**: Use the retriever_tool to find relevant information from your knowledge base. When searching, don't modifiy the user's question to create the query. Just combine the text and visual context to form a comprehensive query. This will help you find more precise solutions.

        4. **Provide comprehensive answers**: Give solutions that address what you see in the image and what the user describes in text.

        TOOLS AVAILABLE:
        - classifer_tool: Classify problems with type and severity (use context from image + text)
        - retriever_tool: Search knowledge base for solutions

        KEY PRINCIPLES:
        - Use visual context to enhance understanding, not just describe images
        - Integrate image information into your classification and search queries
        - Provide more precise help based on complete context
        - Respond in the language used by the user
        - Always cite sources from your knowledge base

        Remember: The image is additional context to help you understand the problem better, not something to simply describe.
        """




        tools_dict = {our_tool.name: our_tool for our_tool in tools}


        def call_llm(state: AgentState) -> AgentState:
            """Function to call the LLM (sans streaming ici)."""
            print("[CALL LLM] Calling LLM with state:", state)
            messages = list(state['messages'])
            
            print("[CALL LLM] Initial messages:", messages)
            
            has_image = False
            image_context = ""
            
            for msg in messages:
                if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
                    for content_item in msg.content:
                        if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                            has_image = True
                            # Extraire le texte qui accompagne l'image pour comprendre le contexte
                            for text_item in msg.content:
                                if isinstance(text_item, dict) and text_item.get("type") == "text":
                                    image_context = text_item.get("text", "")
                            break
            print(f"[CALL LLM] Image détectée: {has_image}")
            if has_image:
                print(f"[CALL LLM] Contexte textuel: {image_context[:100]}...")
                
            # On adapte le prompt selon si la classification est faite ou non
            if state.get("error_classification"):
                # Classification déjà faite, pas d'image
                classification_info = state["error_classification"]
                system_prompt_updated = f"""
                You are PASSE Bot. The error has been classified as:
                - Type: {classification_info['type']}
                - Severity: {classification_info['severity']}

                Now use the retriever tool with user'query to find solutions, citing document sources.
                """
            elif has_image:
                # Image présente - utiliser le prompt avec contexte visuel
                system_prompt_updated = system_prompt_with_visual_context
                print("[CALL LLM] Using visual context prompt")
            else:
                # Pas d'image, utiliser le prompt standard
                system_prompt_updated = system_prompt
                print("[CALL LLM] Using standard prompt")
                
            # On insère en premier
            useful_messages = []
            for m in state["messages"]:
                if isinstance(m, HumanMessage) or isinstance(m, ToolMessage):
                    useful_messages.append(m)
                    
            print("[CALL LLM] Useful messages:", useful_messages)

            # 3. Ajouter le nouveau prompt au début
            messages = [SystemMessage(content=system_prompt_updated)] + useful_messages

            print("[CALL LLM] Messages envoyés au modèle :", messages)
            
            message = llm.invoke(messages)

            print("[CALL LLM] Tool calls:", getattr(message, "tool_calls", None))
            
            print("[CALL LLM] Final state:", state)
            
            return {"messages": [message], "error_classification": state.get("error_classification")}


        def take_action(state: AgentState) -> AgentState:
            """Execute tool calls from the LLM's response."""
            tool_calls = state['messages'][-1].tool_calls
            results = []

            for t in tool_calls:
                tool_name = t['name']
                args = t['args'].get('query', '')

                print(f"\n[TAKE ACTION] Calling Tool: '{tool_name}' with query: {args}")

                if tool_name not in tools_dict:
                    print(f"[TAKE ACTION] Tool '{tool_name}' not found.")
                    result = f"Tool '{tool_name}' not implemented."
                elif tool_name == "classifer_tool":
                    raw = tools_dict[tool_name].invoke(args)
                    clean_json = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
                    try:
                        parsed = json.loads(clean_json)
                        state["error_classification"] = parsed
                        result = parsed
                    except json.JSONDecodeError:
                        print("[TAKE ACTION] Parsing error in classifer_tool output")
                        fallback = {"type": "unknown", "severity": "unknown"}
                        result = fallback
                        state["error_classification"] = fallback
                else:
                    try:
                        result = tools_dict[tool_name].invoke(args)
                    except Exception as e:
                        print(f"[TAKE ACTION] Error calling tool '{tool_name}': {e}")
                        result = f"Error in tool {tool_name}"

                results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
                print(f"[TAKE ACTION] Result from '{tool_name}': {result}")

            # state["messages"].extend(results)
            print("[TAKE ACTION] Final classification:", state.get("error_classification"))
            print("[TAKE ACTION] Messages in state:", len(state["messages"]))
            
            return {"messages": results, "error_classification": state.get("error_classification")}


        graph = StateGraph(AgentState)
        graph.add_node("llm", call_llm)
        graph.add_node("tools", take_action)

        graph.add_conditional_edges(
            "llm",
            should_continue,
            {True: "tools", False: END}
        )
        graph.add_edge("tools", "llm")
        graph.set_entry_point("llm")

        passe_agent = graph.compile()


# 🧪 TEST RAPIDE : Tester l'agent sans interaction
# Utilisez ceci pour voir rapidement si tout fonctionne

# def test_agent():
#     """Test simple de l'agent avec une question prédéfinie"""
#     print("🧪 Test de l'agent en cours...")
    
#     # Chemin vers ton image
#     # image_path = "D:\\RAG_NGSTARS\\faq.jpg"

#     # Encoder l'image
#     # base64_image = encode_image(image_path)
    
#     # Question de test
#     test_question = "Je n'arrive pas à me connecter à mon compte"
    
#     # Créer le message
#     messages = [HumanMessage(
#         content=[
#             {"type": "text", "text": test_question},
#             # {
#             #     "type": "image_url",
#             #     "image_url": {
#             #         "url": f"data:image/jpeg;base64,{base64_image}"
#             #     },
#             # },
#         ]
#     )] # converts back to a HumanMessage type

    
#     print(f"❓ Question de test: {test_question}")
#     print("=" * 50)
    
#     try:
#         # Invoquer l'agent
#         result = passe_agent.invoke({"messages": messages})
        
#         print("\n✅ RÉSULTAT DU TEST:")
#         print("=" * 50)
#         print(result['messages'][-1].content)
        
#         # Vérifier l'état final
#         print(f"\n📊 État final:")
#         print(f"- Nombre total de messages: {len(result['messages'])}")
#         print(f"- Résultat: {result}")
#         return result
#     except Exception as e:
#         print(f"❌ ERREUR lors du test: {e}")
#         import traceback
#         traceback.print_exc()
# # Exécuter le test
# # result = test_agent_simple()