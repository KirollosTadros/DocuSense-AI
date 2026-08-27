FROM my-rag-docker:latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    tk-dev \
    libtk8.6 \
    libtcl8.6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir customtkinter

COPY . .

CMD ["python", "gui_app.py"]