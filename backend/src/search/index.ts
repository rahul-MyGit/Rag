import { RouterQueryEngine, BaseQueryEngine, VectorStoreIndex, LLMSingleSelector, QueryEngineTool, Settings } from 'llamaindex';

/**
 * Create intelligent query router with three engines:
 * 1. Document-only engine (policies, procedures)
 * 2. Transcript-only engine (client conversations)  
 * 3. Cross-reference engine (policies + client evidence)
 */
export function createQueryRouter(
    documentIndex: VectorStoreIndex,
    transcriptIndex: VectorStoreIndex
): RouterQueryEngine {
    const docEngine = documentIndex.asQueryEngine({
        similarityTopK: 5
    });

    const transcriptEngine = transcriptIndex.asQueryEngine({
        similarityTopK: 5
    });

    const crossRefEngine = createCrossReferenceEngine(docEngine, transcriptEngine);

    // Create router using fromDefaults with plain objects
    const router = RouterQueryEngine.fromDefaults({
        queryEngineTools: [
            {
                queryEngine: docEngine,
                description: "Agency policies, procedures, guidelines, and regulations. Use for policy definitions, compliance requirements, and official procedures."
            },
            {
                queryEngine: transcriptEngine,
                description: "Client-case manager conversation transcripts. Use for client behavior analysis, session summaries, and treatment progress."
            },
            {
                queryEngine: crossRefEngine,
                description: "Cross-reference analysis between policies and client interactions. Use for compliance checking, intervention recommendations, and policy application verification."
            }
        ]
    });

    console.log('🎯 Created intelligent query router with 3 engines');
    return router;
}

/**
 * Create cross-reference engine that combines policy and transcript sources
 */
function createCrossReferenceEngine(
    docEngine: BaseQueryEngine, 
    transcriptEngine: BaseQueryEngine
): BaseQueryEngine {
    return {
        query: async (params: any) => {
            const query = typeof params === 'string' ? params : params.query;
            
            // Get policy context first
            const policyResponse = await docEngine.query({ query });
            
            // Use policy context to enhance transcript search
            const enhancedTranscriptQuery = `${query}\n\nPolicy Context: ${policyResponse.toString()}`;
            const transcriptResponse = await transcriptEngine.query({ query: enhancedTranscriptQuery });
            
            // Combine responses
            return {
                response: `Policy Context:\n${policyResponse.toString()}\n\nEvidence from Transcripts:\n${transcriptResponse.toString()}`,
                sourceNodes: []
            };
        }
    } as unknown as BaseQueryEngine;
}
