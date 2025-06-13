import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatOpenAI } from '@langchain/openai';
// import { CohereRerank } from '@langchain/cohere';
import { PineconeStore } from '@langchain/pinecone';
import { ParentDocumentRetriever } from 'langchain/retrievers/parent_document';
import { InMemoryStore } from '@langchain/core/stores';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { CONFIG, ENV } from './index';

export let pinecone: Pinecone;
export let embeddings: OpenAIEmbeddings;
export let llm: ChatOpenAI;
// export let reranker: CohereRerank;
export let docVectorStore: PineconeStore | null = null;
export let transcriptVectorStore: PineconeStore | null = null;

export let docRetriever: ParentDocumentRetriever | null = null;

export let docStore: InMemoryStore | null = null;

export const bm25Storage = {
    documents: new Map<string, any>(),
    transcripts: new Map<string, any>()
};

export async function initializeRAGSystem(): Promise<void> {
    try {
        pinecone = new Pinecone({
            apiKey: ENV.PINECONE_API_KEY!,
        });

        embeddings = new OpenAIEmbeddings({
            openAIApiKey: ENV.OPENAI_API_KEY!,
            modelName: 'text-embedding-ada-002'
        });

        llm = new ChatOpenAI({
            openAIApiKey: ENV.OPENAI_API_KEY!,
            modelName: 'gpt-4o-mini',
            temperature: 0.1
        });

        // reranker = new CohereRerank({
        //     apiKey: ENV.COHERE_API_KEY!,
        //     model: "rerank-v3.5"
        // });

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

        const docIndex = pinecone.index(CONFIG.PINECONE.DOC_INDEX_NAME);
        const transcriptIndex = pinecone.index(CONFIG.PINECONE.TRANSCRIPT_INDEX_NAME);

        docVectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: docIndex,
        });

        transcriptVectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: transcriptIndex,
        });

        docStore = new InMemoryStore();

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

        console.log('Vector stores and Parent Document Retrievers initialized successfully');
    } catch (error) {
        console.error('Error initializing Pinecone indexes:', error);
        throw error;
    }
}