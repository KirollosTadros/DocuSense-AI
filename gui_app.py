from __future__ import annotations
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
from textReader import PDFReader
from textEmbedding import textEmbedding
from vectorStore import VectorStore
from ragPdfAgent import ragPdfAgent
import os

# Set appearance and theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class RagGuiApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("RAG PDF Agent")
        self.geometry("900x700")

        # Initialize backend components
        self.pdf_reader = None
        self.chunk_creator = textEmbedding()
        self.db = VectorStore()
        self.agent = None
        
        # UI State
        self.is_processing = False

        # Create Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="RAG Agent", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.select_file_button = ctk.CTkButton(self.sidebar_frame, text="Browse PDF", command=self.browse_pdf)
        self.select_file_button.grid(row=1, column=0, padx=20, pady=10)

        self.file_label = ctk.CTkLabel(self.sidebar_frame, text="No file selected", wraplength=160, font=ctk.CTkFont(size=12, slant="italic"))
        self.file_label.grid(row=2, column=0, padx=20, pady=5)

        self.project_entry = ctk.CTkEntry(
            self.sidebar_frame,
            justify="center",                      
            font=ctk.CTkFont(family="Helvetica", size=15, weight="bold"),
            height=38,
            corner_radius=10,
            fg_color=("gray85", "#1f2937"),
            border_color=("gray70", "#3b82f6"),
            border_width=2,
            text_color=("gray10", "#f3f4f6"),
        )
        self.project_entry.grid(row=4, column=0, padx=20, pady=(0, 15), sticky="ew")
        self.project_entry.insert(0, "DocuSense-AI")
        self.project_entry.configure(state="readonly")

        self.clear_button = ctk.CTkButton(self.sidebar_frame, text="Clear Chat", fg_color="transparent", border_width=1, command=self.clear_chat)
        self.clear_button.grid(row=5, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Ready", text_color="gray")
        self.status_label.grid(row=6, column=0, padx=20, pady=10)
        
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                               command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 10))
        self.appearance_mode_optionemenu.set("Dark")

        # Main Area
        self.chat_frame = ctk.CTkFrame(self, corner_radius=0)
        self.chat_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)

        self.chat_history = ctk.CTkTextbox(self.chat_frame, state="disabled", wrap="word")
        self.chat_history.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Input Area
        self.input_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.query_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Ask a question about the document...")
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.query_entry.bind("<Return>", lambda e: self.send_query())

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=100, command=self.send_query)
        self.send_button.grid(row=0, column=1)

        self.progress_bar = ctk.CTkProgressBar(self.chat_frame)
        self.progress_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.progress_bar.set(0)

        # Chat tags
        self.chat_history.tag_config("User", foreground="#1f538d")
        self.chat_history.tag_config("Agent", foreground="#2fa572")
        self.chat_history.tag_config("System", foreground="gray")

    def clear_chat(self):
        self.chat_history.configure(state="normal")
        self.chat_history.delete("1.0", tk.END)
        self.chat_history.configure(state="disabled")
        if self.agent:
            self.agent.history = [] 

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def browse_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.clear_chat()
            self.db.reset_collection()
            self.agent = None
            self.file_label.configure(text=os.path.basename(file_path))
            self.process_pdf(file_path)

    def process_pdf(self, file_path):
        self.is_processing = True
        self.status_label.configure(text="Status: Embedding...", text_color="orange")
        self.progress_bar.set(0)
        self.progress_bar.start()
        
        # Run embedding in a thread
        thread = threading.Thread(target=self._embedding_worker, args=(file_path,))
        thread.daemon = True
        thread.start()

    def _embedding_worker(self, file_path):
        try:
            pdf_reader = PDFReader(file_path)
            raw_text = pdf_reader.extract_text()
            
            chunks, embeddings = self.chunk_creator.generateEmbeddings(raw_text)
            self.db.add_chunks(chunks=chunks, embeddings=embeddings)
            
            self.after(0, self._on_embedding_complete)
        except Exception as e:
            self.after(0, lambda: self._on_error(f"Error processing PDF: {str(e)}"))

    def _on_embedding_complete(self):
        self.is_processing = False
        self.status_label.configure(text="Status: Ready", text_color="green")
        self.progress_bar.stop()
        self.progress_bar.set(1)
        self.append_chat("System", "Document processed successfully. You can now ask questions!")

    def send_query(self):
        query = self.query_entry.get()
        if not query or self.is_processing:
            return
        
        self.query_entry.delete(0, tk.END)
        self.append_chat("User", query)
        
        self.status_label.configure(text="Status: Thinking...", text_color="orange")
        self.progress_bar.start()
        
        thread = threading.Thread(target=self._query_worker, args=(query,))
        thread.daemon = True
        thread.start()

    def _query_worker(self, query):
        try:
            if not self.agent:
                self.agent = ragPdfAgent(db=self.db, embedding_model=self.chunk_creator.model)
            
            db_result = self.db.search(query=query, embedding_model=self.chunk_creator.model)
            agent_reply = self.agent.askRagAgent(user_query=query)
            
            self.after(0, lambda: self._on_query_complete(agent_reply))
        except Exception as e:
            error_msg = str(e)
            print(error_msg)
            self.after(0, lambda: self._on_error(f"Error getting response: {error_msg}"))

    def _on_query_complete(self, reply):
        self.status_label.configure(text="Status: Ready", text_color="green")
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.append_chat("Agent", reply)

    def _on_error(self, message):
        self.is_processing = False
        self.status_label.configure(text="Status: Error", text_color="red")
        self.progress_bar.stop()
        messagebox.showerror("Error", message)

    def append_chat(self, sender, message):
        self.chat_history.configure(state="normal")
        self.chat_history.insert(tk.END, f"{sender}: ", (sender,))
        self.chat_history.insert(tk.END, f"{message}\n\n")
        self.chat_history.configure(state="disabled")
        self.chat_history.see(tk.END)

if __name__ == "__main__":
    app = RagGuiApp()
    app.mainloop()
