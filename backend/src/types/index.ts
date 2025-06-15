import type { NodeWithScore } from 'llamaindex';

export interface AgencyMetadata {
    source: string;
    type: 'document' | 'transcript';
    userId?: string;
    clientId?: number; // 1 = nathan, 2 = robert
    fileName: string;
    date?: string; // extracted from filename format: {name}-{month}-{day}
    filePath?: string;
    pageNumber?: number;
}

export interface ChatRequest {
    query: string;
    clientId?: number; // 1 = nathan, 2 = robert
}

export interface SourceInfo {
    fileName: string;
    score: number;
    type: string;
}

export interface QueryResult {
    response: string;
    sourceNodes?: NodeWithScore[];
}