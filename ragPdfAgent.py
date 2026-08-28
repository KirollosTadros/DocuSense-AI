from __future__ import annotations
import os
from typing import List, Optional
from google import genai
from google.genai import types
from vectorStore import VectorStore

class ragPdfAgent:
    def __init__(self, db: VectorStore, embedding_model):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.db = db
        self.embedding_model = embedding_model

        # 1. Define the tool as a clean closure
        def search_db(query: str) -> dict:
            """Searches the vector database for relevant PDF document chunks.

            Args:
                query: The search query used to find matching text chunks.
            """
            print(f"\n---> [TOOL TRIGGERED] Searching DB for: '{query}' <---", flush=True)
            chunks = self.db.search(query=query, embedding_model=self.embedding_model)
            print(f"---> [TOOL FINISHED] Found {len(chunks)} chunks. <---", flush=True)
            return {"chunks": chunks}

        system_instruction = """You are a precise technical AI assistant.

        RULES:
        1. Always search for relevant context using the `search_db` tool before answering technical or factual questions.
        2. Answer the user's question using ONLY the factual context returned by your retrieval tool.
        3. If the retrieved context does not contain the answer, say "I don't know based on the document."
        4. Reference the retrieved context naturally."""

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[search_db],
        )

        self.chat = self.client.chats.create(
            model="gemini-3.1-flash-lite",
            config=config,
        )

    def askRagAgent(self, user_query: str) -> str:
        print(f"\n[AGENT] Sending query: '{user_query}'", flush=True)
        response = self.chat.send_message(user_query)
        return response.text