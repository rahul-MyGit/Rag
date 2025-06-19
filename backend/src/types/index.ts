import { Document } from 'langchain/document';

export interface ChunkMetadata {
    source: string;
    type: 'document' | 'transcript';
    userId?: string;
    clientId?: number; // 1 = nathan, 2 = robert
    parentId?: string;
    chunkIndex: number;
    pageNumber?: number;
    fileName: string;
    summary?: string;
    keywords?: string[];
    timestamp?: string;
    createdAt?: string;
    originalContent?: string;
}

export interface BM25Result {
    document: Document;
    score: number;
}

export interface QueryIntent {
    needsDocuments: boolean;
    needsTranscripts: boolean;
    confidence: number;
    queryType: 'document' | 'transcript' | 'mixed';
}

export interface RetrievalResult {
    content: string;
    confidence: number;
    sources: Array<{
        type: string;
        id: string;
        score: number;
    }>;
    searchStrategy: 'semantic' | 'bm25' | 'hybrid' | 'failed';
}



// export interface PineconeMatch {
//     id: string;
//     score: number;
//     metadata?: {
//         text?: string;
//         type?: string;
//         id?: string;
//     };
// }