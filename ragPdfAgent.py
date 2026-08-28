from __future__ import annotations
import os
from vectorStore import VectorStore
from typing import List, Optional
from google import genai
from google.genai import types

class ragPdfAgent:
    def __init__(self):
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)
            self.history = []
        except Exception as e:
            print(f"Failed to initialize Vertex AI: {e}")
            self.model = None

    def askRagAgent(self, user_query: str, model_name: str = "gemini-3.6-flash", chunks: Optional[List[str]] = None) -> str:    

        context_search = "\n\n".join(chunks) if chunks else "No context provided."

        # Format history for the prompt
        history_str = ""
        for turn in self.history[-5:]:  # Keep last 5 turns for context
            history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        prompt = f"""
        You are a precise technical AI assistant. 
                
        RULES:
        1. Answer the user's question using ONLY the factual context provided below.
        2. If the answer is not in the context, say "I don't know based on the document."
        3. Do not use outside knowledge.
        4. Reference the context naturally.

        CONVERSATION HISTORY:
        {history_str}

        CONTEXT FROM PDF (TOOL OUTPUT):
        {context_search}

        CURRENT USER QUESTION: 
        {user_query}

        ANSWER:
        """

        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Lower temperature for factual precision
                max_output_tokens=1024,
            ),
        )

        assistant_reply = response.text or "I don't know based on the document."

        # Update history
        self.history.append({"user": user_query, "assistant": assistant_reply})

        return assistant_reply