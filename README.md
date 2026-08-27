# RAG PDF Agent with CustomTkinter & Gemini

An intelligent, precise, and visually modern **Retrieval-Augmented Generation (RAG)** application. It features both a desktop Graphical User Interface (GUI) built with CustomTkinter and a Command Line Interface (CLI) fallback. The application allows users to load any local PDF document, partition and generate dense embeddings, store them in a persistent vector database, and converse with the document using Google's latest **Gemini 3.6 Flash** model under strict, factual constraints.

---

## 🚀 What the Application Does

1. **PDF Parsing:** Extracts raw text from any user-selected `.pdf` file.
2. **Text Chunking:** Partitions the raw text into distinct, overlapping word-based chunks to preserve semantic context at boundaries.
3. **Dense Vector Embeddings:** Computes high-dimensional vector representations for each text chunk using a local sentence-transformer.
4. **Persistent Storage:** Saves chunks and embeddings to a localized Chroma database instance (`doc_db`).
5. **Semantic Similarity Retrieval:** Encodes the user's chat query and runs a similarity search to retrieve the top $k$ most relevant context blocks from the vector DB.
6. **Factual Grounded Chat:** Prompts the **Gemini 3.6 Flash** model with the retrieved context and a 5-turn history. It is programmed to answer strictly from the document context or say *"I don't know based on the document"* if the information is unavailable, eliminating hallucinations.

---

## 🏗️ System Architecture & Data Flow

The application follows a clean, decoupled modular design:

```
┌─────────────────┐       ┌────────────────────────┐       ┌─────────────────┐
│  User Interface │       │     textReader.py      │       │ textEmbedding.py│
│  (GUI / CLI)    ├──────►│ (Extracts text via PDF)├──────►│(Word Chunker &  │
└──────┬▲─────────┘       └────────────────────────┘       │Transformer Embed)
       ││                                                          │
       ││                                                          ▼
       ││                                                  ┌─────────────────┐
       ││                                                  │  vectorStore.py │
       ││                                                  │(Persists vectors│
       ││                                                  │  in Chroma DB)  │
       ││                                                  └───────┬─────────┘
       ││                                                          │
       ││                 ┌────────────────────────┐               │
       │└─────────────────┤     ragPdfAgent.py     │◄──────────────┘
       └─────────────────►│ (Google GenAI SDK chat)│ (Retrieves top K matches)
                          └────────────────────────┘
```

### Module Breakdown:
*   **`textReader.py` (PDF Parser):** Employs `pypdf` to extract raw textual data page-by-page from a target PDF document path.
*   **`textEmbedding.py` (Chunker & Embedder):** 
    *   **Word Chunker:** Implements a sliding window algorithm (`wordChunker`) that tokenizes the text by spaces, generating chunks of $300$ words with an overlap of $50$ words to retain relational cohesion.
    *   **Embedding Generator:** Feeds chunks into the Hugging Face `all-MiniLM-L6-v2` Sentence Transformer model to produce $384$-dimensional dense vector embeddings.
*   **`vectorStore.py` (Vector DB):** Interacts with `chromadb` using a `PersistentClient` targeting `./doc_db`. Handles database recreation/reset, indexing, and executing cosine similarity-based querying (retrieving the top $3$ matches).
*   **`ragPdfAgent.py` (Agentic Chat Client):** Leverages Google's official new `google-genai` client library. Constructs a prompt injection holding context, conversation history, and user input. It issues requests to `gemini-3.6-flash` with a temperature of `0.1` to maximize precision and reduce creativity.
*   **`gui_app.py` & `app.py` (Frontend Interfaces):**
    *   `gui_app.py`: Provides an interactive dark/light themed window using `customtkinter` with multi-threaded execution so that the GUI never freezes during heavy embedding or inference operations.
    *   `app.py`: Standard terminal CLI variant.

---

## 🛠️ Technology Stack

*   **Language:** Python 3.10
*   **UI Framework:** CustomTkinter & Tkinter (cross-platform graphical interface wrapper)
*   **PDF Library:** `pypdf`
*   **Vector Database:** `chromadb` (Chroma Vector Database Engine)
*   **Embedding Model:** `sentence-transformers` (`all-MiniLM-L6-v2` / 384 dimensions)
*   **Large Language Model Platform:** Google GenAI SDK (`google-genai` Python library)
*   **LLM Model:** `gemini-3.6-flash`
*   **Containerization:** Docker (multi-stage setup with optimized local caching)

---

## 🐳 Docker Setup Steps

This application relies on a dual-stage Docker configuration to separate general dependencies from specific application layers, speeding up builds and managing dependencies cleanly.

### Step 1: Build the Base Image
The base image installs compiler toolchains, pulls standard packages from `requirements.txt`, and leverages a local offline pip cache folder (`.pip-cache/`) to perform fast installations without fetching from the web.

Run the following command in your terminal:
```bash
docker build -f Dockerfile.base -t my-rag-docker:latest .
```

### Step 2: Build the Main/Application Image
The main application image builds **on top** of the base image. It configures Tkinter system packages (required for custom GUI desktop applications inside containers) and installs `customtkinter` before injecting the source files.

Run the following command:
```bash
docker build -f Dockerfile -t my-rag-app:latest .
```

---

## 🔑 Exporting the Gemini API Key

Before running the application, you must configure your **Google Gemini API Key**. The application uses this environment variable to authenticate the `google-genai` SDK and perform queries with `gemini-3.6-flash`.

On your host terminal, run:
```bash
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

*Note: Ensure this key is set before launching the Docker container, as the container reads the key directly from your terminal environment.*

---

## 🏃 Running the Application

To execute the GUI application within your system using Docker, you need to share graphical and file resources between your host system and the running container.

Run the container using the command below:

```bash
docker run --rm -it \
   --net=host \
   --ipc=host \
   --user $(id -u):$(id -g) \
   -e DISPLAY=$DISPLAY \
   -e GEMINI_API_KEY="$GEMINI_API_KEY" \
   -e HF_HOME=/app/.hf_cache \
   -e SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache \
   -e TORCH_HOME=/app/.hf_cache \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v "$HOME:$HOME" \
   -v "$(pwd):/app" \
   -w /app \
   my-rag-app:latest
```

### ⚙️ Deep Dive: Understanding the Run Command Flags

| Flag | Category | Purpose & Description |
| :--- | :--- | :--- |
| `--rm` | Container State | Automatically cleans up and deletes the container instance immediately upon exit, saving host disk space. |
| `-it` | Terminal Mode | Combines `-i` (interactive, keeps STDIN open) and `-t` (allocates a pseudo-TTY) to display runtime console outputs in real-time. |
| `--net=host` | Networking | Bypasses container network isolation, binding the container directly to your host's network interfaces. |
| `--ipc=host` | Inter-Process | Shares the host’s IPC namespace. This is critical for high-performance X11 shared memory (MIT-SHM) to allow GUI windows to render smoothly. |
| `--user $(id -u):$(id -g)`| Permissions | Forces the container to execute using your host user's exact User ID and Group ID instead of `root`. This guarantees that any generated database folders (`doc_db`) or caches on the host have the correct user permissions. |
| `-e DISPLAY=$DISPLAY` | Graphical Display | Passes your active X11 GUI server session address (`$DISPLAY`) to the container so Tkinter knows where to open the app window. |
| `-e GEMINI_API_KEY="..."` | Configuration | Forwards your API key from the host shell into the container so the backend GenAI SDK can authenticate. |
| `-e HF_HOME=...`<br>`-e SENTENCE_TRANSFORMERS_HOME=...`<br>`-e TORCH_HOME=...` | Model Caching | Standardizes model search directories to `/app/.hf_cache`. By anchoring this folder to the container, models downloaded by PyTorch or Hugging Face are cached directly in your project folder, preventing redownloads. |
| `-v /tmp/.X11-unix:/tmp/.X11-unix` | GUI Rendering Volume | Mounts the local Unix graphical server communications socket into the container, allowing CustomTkinter GUI drawings to output on your physical monitor. |
| `-v "$HOME:$HOME"` | Directory Volume | Mounts your entire host home directory onto the exact same path inside the container. This lets you use the app’s "Browse PDF" button to select files from anywhere in your host user folders. |
| `-v "$(pwd):/app"` | Workdir Volume | Mounts your current project repository folder to `/app` inside the container, persisting databases and configurations to your physical machine. |
| `-w /app` | Directory State | Sets `/app` as the active execution context for commands run inside the container. |
| `my-rag-app:latest` | Target | Specifying the built application Docker image to boot. |

---

## 💻 Optional: Running Locally without Docker

If you have a Python 3.10+ environment set up on your machine and wish to run without Docker, follow these commands:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install customtkinter
   ```
2. **Export API Key:**
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```
3. **Run GUI App:**
   ```bash
   python gui_app.py
   ```
4. **Run CLI Fallback App:**
   ```bash
   python app.py
   ```
