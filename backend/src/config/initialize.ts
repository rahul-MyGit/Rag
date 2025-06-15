// Clean LlamaIndex initialization orchestrator
import { Pinecone } from '@pinecone-database/pinecone';
import { RouterQueryEngine, Settings } from 'llamaindex';
import { OpenAI, OpenAIEmbedding } from '@llamaindex/openai';
import { CONFIG, ENV } from './index';
import { buildDocumentIndex, buildTranscriptIndex } from '../main/ingestion';
import { createQueryRouter } from '../search';

// Global instances
export let pinecone: Pinecone;
export let llm: OpenAI;
export let embedModel: OpenAIEmbedding;
export let agencyRouter: RouterQueryEngine;

export async function initializeRAGSystem(): Promise<void> {
    try {
        console.log('🚀 Initializing LlamaIndex RAG system...');

        // Initialize clients
        await initializeClients();
        
        // Initialize Pinecone indexes
        await initializePineconeIndexes();

        // Build document and transcript indexes
        console.log('📚 Building document and transcript indexes...');
        const [documentIndex, transcriptIndex] = await Promise.all([
            buildDocumentIndex(),
            buildTranscriptIndex()
        ]);

        if (!documentIndex || !transcriptIndex) {
            throw new Error('Failed to build one or more indexes');
        }

        // Create intelligent query router
        agencyRouter = createQueryRouter(documentIndex, transcriptIndex);
        
        console.log('✅ LlamaIndex RAG system initialized successfully');
        console.log('📊 System ready for queries');
        
    } catch (error) {
        console.error('❌ Error initializing RAG system:', error);
        throw error;
    }
}

/**
 * Initialize external service clients
 */
async function initializeClients(): Promise<void> {
    // Initialize Pinecone
    pinecone = new Pinecone({
        apiKey: ENV.PINECONE_API_KEY!,
    });
    console.log('📌 Pinecone client initialized');

    // Initialize OpenAI
    llm = new OpenAI({
        apiKey: ENV.OPENAI_API_KEY!,
        model: 'gpt-4o-mini',
        temperature: 0.1
    });

    embedModel = new OpenAIEmbedding({
        apiKey: ENV.OPENAI_API_KEY!,
        model: 'text-embedding-ada-002'
    });

    // Set global settings
    Settings.llm = llm;
    Settings.embedModel = embedModel;
    
    console.log('🤖 OpenAI client initialized');

    // Test connections
    await testConnections();
}

/**
 * Test external service connections
 */
async function testConnections(): Promise<void> {
    try {
        // Test Pinecone connection
        await pinecone.listIndexes();
        console.log('✅ Pinecone connection verified');

        // OpenAI connection will be tested when first used
        console.log('✅ OpenAI connection configured');
        
    } catch (error) {
        console.error('❌ Connection test failed:', error);
        throw error;
    }
}

/**
 * Initialize Pinecone indexes (create if they don't exist)
 */
async function initializePineconeIndexes(): Promise<void> {
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
        console.log(`✅ Created Pinecone index: ${CONFIG.PINECONE.DOC_INDEX_NAME}`);
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
        console.log(`✅ Created Pinecone index: ${CONFIG.PINECONE.TRANSCRIPT_INDEX_NAME}`);
    }
}
