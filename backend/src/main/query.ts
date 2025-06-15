import { RouterQueryEngine, BaseQueryEngine, VectorStoreIndex } from 'llamaindex';
import type { ChatRequest, QueryResult, SourceInfo } from '../types';

/**
 * Process query with client filtering for transcripts
 */
export async function processQuery(
    request: ChatRequest,
    router: RouterQueryEngine
): Promise<QueryResult> {
    const { query, clientId } = request;
    
    console.log(`🔍 Processing query: "${query}" for client ${clientId || 'general'}`);
    
    // Add client context if specified
    let enhancedQuery = query;
    if (clientId) {
        const clientName = clientId === 1 ? 'Nathan' : clientId === 2 ? 'Robert' : 'unknown';
        enhancedQuery = `Query about client: ${clientName} (ID: ${clientId})\n\n${query}`;
    }
    
    const result = await router.query({ query: enhancedQuery });
    
    return {
        response: result.response,
        sourceNodes: result.sourceNodes
    };
}

/**
 * Create client-specific query engine that filters transcripts by clientId
 */
export function createClientQueryEngine(
    transcriptIndex: VectorStoreIndex,
    clientId: number
): BaseQueryEngine {
    return transcriptIndex.asQueryEngine({
        similarityTopK: 5,
        preFilters: {
            filters: [
                {
                    key: "clientId",
                    value: clientId,
                    operator: "=="
                }
            ]
        }
    });
}

/**
 * Extract source information from query results
 */
export function extractSourceInfo(sourceNodes: any[]): SourceInfo[] {
    if (!sourceNodes) return [];
    
    return sourceNodes.map(node => ({
        fileName: node.metadata?.fileName || node.metadata?.source || 'Unknown',
        score: node.score || 0,
        type: node.metadata?.type || 'document'
    }));
}