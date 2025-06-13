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
}

export interface IntentAnalysis {
    needsDocuments: boolean;
    needsTranscripts: boolean;
    userId?: string;
    confidence: number;
    queryType: 'factual' | 'conversational' | 'analytical' | 'mixed';
}

export interface BM25Result {
    document: Document;
    score: number;
}

export interface HybridSearchResult {
    document: Document;
    semanticScore: number;
    bm25Score: number;
    combinedScore: number;
}

export interface RetrievalResult {
    chunks: Document[];
    sources: string[];
    confidence: number;
    searchStrategy: string;
}