import dotenv from 'dotenv';

dotenv.config();

export const CONFIG = {
    PINECONE: {
        DOC_INDEX_NAME: 'documents-index-2',
        TRANSCRIPT_INDEX_NAME: 'transcripts-index-2',
        DIMENSION: 1536,
        ENVIRONMENT: 'us-east-1'
    },
    PATHS: {
        DOCS_DIR: process.env.DOCS_DIR || 'D:\\100xSuper30\\rag-playground\\docs_for_test',
        TRANSCRIPTS_DIR: process.env.TRANSCRIPTS_DIR || 'D:\\100xSuper30\\rag-playground\\transcriptions_for_test'
    }
};

export const ENV = {
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    PINECONE_API_KEY: process.env.PINECONE_API_KEY,
    PINECONE_ENVIRONMENT: process.env.PINECONE_ENVIRONMENT,
    PORT: process.env.PORT || 3000
}