# MyRAG

### Scientific Research Project on the topic of RAG pipeline development.

### Key features of the project:

* Parsing a EPUB document to create an information source from PostgreSQL documentation ("./data/17.4-ru.epub").

* The project is presented in the form of a micro-service architecture consisting of: \
    QWEN3-1.7B (generating response on prompt after RAG) \
    BGE (text embddeing) \
    BGE-Reranker (reranker for choosing the best candidates from all) \
    PostgreSQL-DB (docker with db for storaging parsed data - text-chunks, uris, etc)

* Benchmark - measures accuracy on how RAG matches the desired retrieval information to the prompt.

### Description of the benchmark's work: 
The dataset was collected where each element is text and 2 relevant prompts to it. \
Then iterating over the dataset RAG extracts the top-k texts for the prompt and then checks whether the original text is included in this top.

Benchmark's results are in "benchmark_results.txt" or "benchmark_visual.png".

### Instructions for launching the project locally:
```
# Firstly start docker
# Stop or remove existing container

# sudo docker stop docs
# sudo docker rm docs
sudo docker run --name docs -p 5432:5432 -e POSTGRES_PASSWORD=123456 nerdspot/pgvecto.rs:pg17-v0.4.0-1

# Connect to db and make database "docs"
psql -h localhost -p 5432 -U postgres

# Prepare data for RAG: 
python3 process_text_to_db.py

# Start all services:
python3 src/service_lm.py
python3 src/service_bge.py
python3 src/service_reranker_bge.py

# Start app:
python3 app.py
```



