import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatOpenAI } from '@langchain/openai';
import { CohereRerank } from '@langchain/cohere';
import { PineconeStore } from '@langchain/pinecone';
import { ParentDocumentRetriever } from 'langchain/retrievers/parent_document';
import { InMemoryStore } from '@langchain/core/stores';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { CONFIG } from './index';

// Global instances
export let pinecone: Pinecone;
export let embeddings: OpenAIEmbeddings;
export let llm: ChatOpenAI;
export let reranker: CohereRerank;
export let docVectorStore: PineconeStore | null = null;
export let transcriptVectorStore: PineconeStore | null = null;

// Parent Document Retrievers - LangChain's official solution
export let docRetriever: ParentDocumentRetriever | null = null;
export let transcriptRetriever: ParentDocumentRetriever | null = null;

// Document stores for parent chunks
export let docStore: InMemoryStore | null = null;
export let transcriptStore: InMemoryStore | null = null;

// BM25 storage (in production, use Redis or similar)
export const bm25Storage = {
    documents: new Map<string, any>(),
    transcripts: new Map<string, any>()
};

export async function initializeRAGSystem(): Promise<void> {
    try {
        // Initialize clients
        pinecone = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY!,
        });

        embeddings = new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_API_KEY!,
            modelName: 'text-embedding-ada-002'
        });

        llm = new ChatOpenAI({
            openAIApiKey: process.env.OPENAI_API_KEY!,
            modelName: 'gpt-4',
            temperature: 0.1
        });

        reranker = new CohereRerank({
            apiKey: process.env.COHERE_API_KEY!,
            model: 'rerank-english-v2.0'
        });

        // Initialize indexes
        await initializePineconeIndexes();

        console.log('RAG System initialized successfully');
    } catch (error) {
        console.error('Error initializing RAG system:', error);
        throw error;
    }
}

async function initializePineconeIndexes(): Promise<void> {
    try {
        const existingIndexes = await pinecone.listIndexes();

        // Create document index if not exists
        if (!existingIndexes.indexes?.find(idx => idx.name === CONFIG.PINECONE.DOC_INDEX_NAME)) {
            await pinecone.createIndex({
                name: CONFIG.PINECONE.DOC_INDEX_NAME,
                dimension: CONFIG.PINECONE.DIMENSION,
                metric: 'cosine',
                spec: {
                    serverless: {
                        cloud: 'aws',
                        region: CONFIG.PINECONE.ENVIRONMENT
                    }
                }
            });
            console.log(`Created index: ${CONFIG.PINECONE.DOC_INDEX_NAME}`);
        }

        // Create transcript index if not exists
        if (!existingIndexes.indexes?.find(idx => idx.name === CONFIG.PINECONE.TRANSCRIPT_INDEX_NAME)) {
            await pinecone.createIndex({
                name: CONFIG.PINECONE.TRANSCRIPT_INDEX_NAME,
                dimension: CONFIG.PINECONE.DIMENSION,
                metric: 'cosine',
                spec: {
                    serverless: {
                        cloud: 'aws',
                        region: CONFIG.PINECONE.ENVIRONMENT
                    }
                }
            });
            console.log(`Created index: ${CONFIG.PINECONE.TRANSCRIPT_INDEX_NAME}`);
        }

        // Initialize vector stores
        const docIndex = pinecone.index(CONFIG.PINECONE.DOC_INDEX_NAME);
        const transcriptIndex = pinecone.index(CONFIG.PINECONE.TRANSCRIPT_INDEX_NAME);

        docVectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: docIndex,
        });

        transcriptVectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: transcriptIndex,
        });

        // Initialize document stores for parent chunks
        docStore = new InMemoryStore();
        transcriptStore = new InMemoryStore();

        // Initialize ParentDocumentRetrievers
        docRetriever = new ParentDocumentRetriever({
            vectorstore: docVectorStore,
            docstore: docStore,
            parentSplitter: new RecursiveCharacterTextSplitter({
                chunkSize: CONFIG.CHUNKING.DOCUMENT.PARENT_SIZE,
                chunkOverlap: CONFIG.CHUNKING.DOCUMENT.PARENT_OVERLAP,
                separators: ['\n\n', '\n', '. ', '; ', ', ', ' ', '']
            }),
            childSplitter: new RecursiveCharacterTextSplitter({
                chunkSize: CONFIG.CHUNKING.DOCUMENT.CHILD_SIZE,
                chunkOverlap: CONFIG.CHUNKING.DOCUMENT.CHILD_OVERLAP,
                separators: ['\n\n', '\n', '. ', ' ', '']
            }),
            childK: 20,
            parentK: 5
        });

        transcriptRetriever = new ParentDocumentRetriever({
            vectorstore: transcriptVectorStore,
            docstore: transcriptStore,
            parentSplitter: new RecursiveCharacterTextSplitter({
                chunkSize: CONFIG.CHUNKING.TRANSCRIPT.PARENT_SIZE,
                chunkOverlap: CONFIG.CHUNKING.TRANSCRIPT.PARENT_OVERLAP,
                separators: ['\n\n', '\n', '. ', '? ', '! ', ', ', ' ', '']
            }),
            childSplitter: new RecursiveCharacterTextSplitter({
                chunkSize: CONFIG.CHUNKING.TRANSCRIPT.CHILD_SIZE,
                chunkOverlap: CONFIG.CHUNKING.TRANSCRIPT.CHILD_OVERLAP,
                separators: ['\n', '. ', '? ', '! ', ' ', '']
            }),
            childK: 20,
            parentK: 5
        });

        console.log('Vector stores and Parent Document Retrievers initialized successfully');
    } catch (error) {
        console.error('Error initializing Pinecone indexes:', error);
        throw error;
    }
}