export const CONFIG = {
    PINECONE: {
        DOC_INDEX_NAME: 'documents-index',
        TRANSCRIPT_INDEX_NAME: 'transcripts-index',
        DIMENSION: 1536,
        ENVIRONMENT: 'us-east-1'
    },
    CHUNKING: {
        DOCUMENT: {
            PARENT_SIZE: 1000,
            PARENT_OVERLAP: 200,
            CHILD_SIZE: 250,
            CHILD_OVERLAP: 50
        },
        TRANSCRIPT: {
            PARENT_SIZE: 1000,
            PARENT_OVERLAP: 200,
            CHILD_SIZE: 250,
            CHILD_OVERLAP: 50
        }
    },
    SEARCH: {
        SEMANTIC_WEIGHT: 0.7,
        BM25_WEIGHT: 0.3,
        TOP_K: 20,
        RERANK_TOP_K: 5
    }
};