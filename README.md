# RAG

- Play and Learn

## Steps Performing :
- Embedding the documents and transcript of user in 2 different store

- INGESTION: doc  → parse pdf → parent chunk → child chunk -> embedding → index.
             transcription  -> parse pdf -> chunk -> embedding -> index

- RETRIEVAL: analyze question intent → Embedding → Hybrid Search → rerank chunks → Verification Layer → send to LLM with original user query to generate response
             - If llm said NO then rephrase the question and pass it to embedding process again
             - If llm said YES then pass context + prompt + question to llm and stream the response
