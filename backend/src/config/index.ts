import dotenv from 'dotenv';

dotenv.config();

export const CONFIG = {
    PINECONE: {
        DOC_INDEX_NAME: 'documents-index',
        TRANSCRIPT_INDEX_NAME: 'transcripts-index',
        DIMENSION: 1536,
        ENVIRONMENT: 'us-east-1'
    },
    CHUNKING: {
        DOCUMENT: {
            PARENT_SIZE: 2000,
            PARENT_OVERLAP: 200,
            CHILD_SIZE: 900,
            CHILD_OVERLAP: 100
        },
        TRANSCRIPT: {
            PARENT_SIZE: 1000,
            PARENT_OVERLAP: 200,
            CHILD_SIZE: 250,
            CHILD_OVERLAP: 50
        }
    },
    SEARCH: {
        SEMANTIC_WEIGHT: 0.5,
        BM25_WEIGHT: 0.5,
        TOP_K: 20,
        RERANK_TOP_K: 5
    }
};

export const ENV = {
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    PINECONE_API_KEY: process.env.PINECONE_API_KEY,
    PINECONE_ENVIRONMENT: process.env.PINECONE_ENVIRONMENT,
    COHERE_API_KEY: process.env.COHERE_API_KEY,
    PORT: process.env.PORT || 3000,
    DOC_INDEX_NAME: process.env.DOC_INDEX_NAME || 'documents-index',
    TRANSCRIPT_INDEX_NAME: process.env.TRANSCRIPT_INDEX_NAME || 'transcripts-index',
    DIMENSION: process.env.DIMENSION || 1536
}