# Production RAG

Planned architecture:

ingestion/
  parser.py
  metadata_pipeline.py
  chunker.py
  knowledge_graph.py

retrieval/
  qdrant_store.py
  retriever.py
  query_expansion.py
  hybrid_search.py

ranking/
  reranker.py

llm/
  qwen.py

app.py
