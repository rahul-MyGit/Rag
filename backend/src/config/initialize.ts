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
// Dense vector stores
export let docDenseVectorStore: PineconeStore | null = null;
export let transcriptDenseVectorStore: PineconeStore | null = null;

// Sparse vector stores (we'll use direct Pinecone client for these)
export let docSparseIndex: any = null;
export let transcriptSparseIndex: any = null;
export let docDenseIndex: any = null;
export let transcriptDenseIndex: any = null;

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
        
        // Create indexes if they don't exist
        const indexesToCreate = [
            { name: CONFIG.PINECONE.DOC_DENSE_INDEX_NAME, type: 'dense' },
            { name: CONFIG.PINECONE.DOC_SPARSE_INDEX_NAME, type: 'sparse' },
            { name: CONFIG.PINECONE.TRANSCRIPT_DENSE_INDEX_NAME, type: 'dense' },
            { name: CONFIG.PINECONE.TRANSCRIPT_SPARSE_INDEX_NAME, type: 'sparse' }
        ];

        for (const { name, type } of indexesToCreate) {
            if (!existingIndexes.indexes?.find(idx => idx.name === name)) {
                const indexConfig: any = {
                    name,
                    dimension: type === 'dense' ? CONFIG.PINECONE.DIMENSION : 10000, // Max 20k for sparse, using 10k for safety
                    metric: 'cosine',
                    spec: {
                        serverless: {
                            cloud: 'aws',
                            region: CONFIG.PINECONE.ENVIRONMENT
                        }
                    }
                };

                await pinecone.createIndex(indexConfig);
                console.log(`Created ${type} index: ${name}`);
                
                // Wait a bit for index to be ready
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }

        // Initialize index references
        docDenseIndex = pinecone.index(CONFIG.PINECONE.DOC_DENSE_INDEX_NAME);
        docSparseIndex = pinecone.index(CONFIG.PINECONE.DOC_SPARSE_INDEX_NAME);
        transcriptDenseIndex = pinecone.index(CONFIG.PINECONE.TRANSCRIPT_DENSE_INDEX_NAME);
        transcriptSparseIndex = pinecone.index(CONFIG.PINECONE.TRANSCRIPT_SPARSE_INDEX_NAME);

        // Initialize LangChain vector stores for dense embeddings
        docDenseVectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: docDenseIndex,
        });

        transcriptDenseVectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex: transcriptDenseIndex,
        });

        docStore = new InMemoryStore();

        docRetriever = new ParentDocumentRetriever({
            vectorstore: docDenseVectorStore,
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

        console.log('✅ All 4 Pinecone indexes initialized successfully:');
        console.log(`  📚 Documents Dense: ${CONFIG.PINECONE.DOC_DENSE_INDEX_NAME}`);
        console.log(`  📊 Documents Sparse: ${CONFIG.PINECONE.DOC_SPARSE_INDEX_NAME}`);
        console.log(`  💬 Transcripts Dense: ${CONFIG.PINECONE.TRANSCRIPT_DENSE_INDEX_NAME}`);
        console.log(`  📈 Transcripts Sparse: ${CONFIG.PINECONE.TRANSCRIPT_SPARSE_INDEX_NAME}`);
    } catch (error) {
        console.error('Error initializing Pinecone indexes:', error);
        throw error;
    }
}