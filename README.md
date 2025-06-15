# RAG

- Play and Learn

## Steps Performing :
- Embedding the documents and transcript of user in 2 different store

- INGESTION: doc  → parse pdf → chunk -> embedding → pinecone index.
             transcription  -> parse pdf -> chunk -> embedding ->pinecone index

- RETRIEVAL: (PREV) Lanchain - analyze question intent → Embedding → Hybrid Search → rerank chunks → Verification Layer → send to LLM with original user query to generate response
             - If llm said NO then rephrase the question and pass it to embedding process again
             - If llm said YES then pass context + prompt + question to llm and stream the response

             (NOW) LlamaIndex - handle all the things with native support


## Current Project Structure:

```
rag-playground/
├── backend/                 # Node.js + LlamaIndex API
│   ├── src/
│   │   ├── config/         # Environment & Pinecone setup
│   │   ├── main/           # Core RAG logic
│   │   │   ├── ingestion.ts    # Document/transcript indexing
│   │   │   └── query.ts        # Query processing
│   │   ├── search/         # Query routing & engines
│   │   ├── utils/          # Document loading utilities
│   │   ├── types/          # TypeScript interfaces
│   │   └── index.ts        # Express server
│   └── package.json
├── frontend/               # Next.js React UI
│   ├── src/app/
│   │   ├── page.tsx        # Main chat interface
│   │   |── layout.tsx      # App layout
│   │   
│   └── package.json
└── README.md
```

