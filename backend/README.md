# Agency RAG Backend (LlamaIndex)

A LlamaIndex-powered RAG system for agency policy documents and client transcripts.

## Features

- **Smart Query Routing**: Automatically routes queries to appropriate sources (documents/transcripts/cross-reference)
- **Streaming Responses**: Real-time response streaming via Server-Sent Events
- **Client Context**: Supports Nathan (ID: 1) and Robert (ID: 2) client filtering
- **Nested Directory Support**: Loads transcripts from `/nathan` and `/robert` subdirectories
- **Date Metadata**: Extracts dates from filenames (`{name}-{month}-{day}.pdf`)

## Setup

1. Install dependencies:
```bash
bun install
```

2. Create `.env` file:
```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1
```

3. Ensure document structure:
```
/docs_for_test/*.pdf                    # Policy documents
/transcriptions_for_test/nathan/*.pdf   # Nathan's transcripts
/transcriptions_for_test/robert/*.pdf   # Robert's transcripts
```

## Running

```bash
bun run dev
```

Server runs on http://localhost:3001

## API

### POST /chat
Streaming chat endpoint with SSE response.

**Request:**
```json
{
  "query": "Your question here",
  "clientId": 1  // Optional: 1=Nathan, 2=Robert
}
```

**Response:** Server-Sent Events stream with:
- `chunk`: Response text chunks
- `metadata`: Source information
- `done`: Stream completion signal
- `error`: Error messages

## Architecture

- **LlamaIndex RouterQueryEngine**: Auto-routes queries intelligently
- **Pinecone Vector Store**: Separate indexes for documents and transcripts
- **Cross-Reference Engine**: Combines policy context with client evidence
- **Built-in Relevance**: No manual verification needed
