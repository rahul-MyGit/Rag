# RAG

- Play and Learn

## Steps Performing :
- Embedding the documents and transcript of user in 2 different store

- INGESTION: doc / transcription → parse pdf (if doc) → chunk → summary → embedding → index. 

- RETRIEVAL: analyze question intent → Embedding → Hybrid Search → rerank chunks → Verification Layer → send to LLM with original user query to generate response