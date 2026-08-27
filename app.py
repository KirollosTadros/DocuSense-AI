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

while True:
    user_query = input("Please Enter your question or quit to close: ")

    if user_query == "quit":
        break

    db_result = db.search(query=user_query, embedding_model=myChunkCreator.model)


    agent = ragPdfAgent()

    agent_reply = agent.askRagAgent(user_query=user_query, chunks=db_result)

    print(agent_reply)