from __future__ import annotations
from textReader import PDFReader
from textEmbedding import textEmbedding
from vectorStore import VectorStore
from ragPdfAgent import ragPdfAgent

pdf_path=input("Enter Pdf path: ")
pdfReader = PDFReader(pdf_path)

raw_text = pdfReader.extract_text()

myChunkCreator = textEmbedding()
chunks, embeddings = myChunkCreator.generateEmbeddings(raw_text)

db = VectorStore()
db.add_chunks(chunks=chunks, embeddings=embeddings)

agent = ragPdfAgent(db=db, embedding_model=myChunkCreator.model)

while True:
    user_query = input("Please Enter your question or quit to close: ")

    if user_query == "quit":
        break

    agent_reply = agent.askRagAgent(user_query=user_query)

    print(agent_reply)