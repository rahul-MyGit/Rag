import { retrieveContext } from "./retrieval";
import { generateResponse } from "./response";
import { type RetrievalResult } from "../types";

/**
 * RAG Query Pipeline flow:
 * 
 * RETRIEVAL: analyze question intent → Embedding → Hybrid Search 
 * (documents: whole vectorDB | transcripts: user-specific chunks) → 
 * rerank chunks → send to LLM with user query to generate response
 */

export async function processQuery(query: string, userId: string): Promise<{
    answer: string;
    retrievalResult: RetrievalResult;
    processingSteps: string[];
}> {
    const processingSteps: string[] = [];
    
    try {
        processingSteps.push("🔍 Starting RAG Query Pipeline");
        processingSteps.push(`📝 User Query: "${query}"`);
        
        if (userId) {
            processingSteps.push(`👤 User Context: ${userId}`);
        }

        processingSteps.push("🚀 Executing retrieval pipeline:");
        processingSteps.push("  Step 1: Analyze question intent");
        processingSteps.push("  Step 2: Generate query embeddings");
        processingSteps.push("  Step 3: Determine user context"); 
        processingSteps.push("  Step 4: Execute agentic retrieval strategy");
        processingSteps.push("  Step 5: Generate response");

        const retrievalResult = await retrieveContext(query, userId);
        
        if (retrievalResult.content) {
            processingSteps.push(`✅ Retrieval successful: Content found`);
            processingSteps.push(`📚 Sources: ${retrievalResult.sources.map(s => s.id).join(', ')}`);
            processingSteps.push(`🎯 Strategy: ${retrievalResult.searchStrategy}`);
            processingSteps.push(`📊 Confidence: ${(retrievalResult.confidence * 100).toFixed(1)}%`);
        } else {
            processingSteps.push("❌ No relevant content found");
        }

        processingSteps.push("🤖 Generating response with LLM...");
        const answer = await generateResponse(query, retrievalResult);
        processingSteps.push("✅ Response generated successfully");

        return {
            answer,
            retrievalResult,
            processingSteps
        };

    } catch (error) {
        processingSteps.push(`❌ Error in query processing: ${error}`);
        console.error('Error in query processing:', error);
        
        return {
            answer: "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question.",
            retrievalResult: {
                content: '',
                sources: [],
                confidence: 0,
                searchStrategy: 'failed'
            },
            processingSteps
        };
    }
}
