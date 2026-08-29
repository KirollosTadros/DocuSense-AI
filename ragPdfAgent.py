from __future__ import annotations
import os
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Part, Tool
from vectorStore import VectorStore

class ragPdfAgent:
    def __init__(self, db: VectorStore, embedding_model):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.db = db
        self.embedding_model = embedding_model

        self.search_db = FunctionDeclaration(
            name="search_db",
            description="Get Information needed from the database",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "query to search for"}
                },
            },
        )

        self.pdf_tool = Tool(
            function_declarations=[
                self.search_db
            ],
        )
    

        system_instruction = """You are a precise technical AI assistant.

        RULES:
        1. The document you need is stored using Text Embedding LLM model in a vector database.
        2. To retrieve data from the vector databse use tool 'pdf_tool' before answering question to get chunks.
        3. Answer the user's question using the factual context returned by your retrieval tool from the vector database.
        4. If the retrieved context does not contain the answer, say "I don't know based on the document."
        5. Answer the used question with explaination based on the data retrieved"""

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            tools=[self.pdf_tool],
        )

        self.chat = self.client.chats.create(
            model="gemini-3.1-flash-lite",
            config=config,
        )
    def db_retrieval(self, query: str) -> dict:
        return {"chunks":self.db.search(query = query, embedding_model = self.embedding_model)}

    def askRagAgent(self, user_query: str) -> str:
        response = self.chat.send_message(user_query)
        
        while response.function_calls:
            function_name = response.function_calls[0].name
            query = response.function_calls[0].args.get('query')
            chunks = self.db_retrieval(query)
            function_response = [
                types.Part.from_function_response(
                    name = function_name,
                    response={"content": chunks}
                )
            ]

            response = self.chat.send_message(function_response)

        return response.text