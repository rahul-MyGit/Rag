import { retrieveContext } from "./retrieval";
import { generateResponse } from "./response";
import { type RetrievalResult } from "../types";

/**
 * Complete RAG Query Pipeline implementing the specified flow:
 * 
 * RETRIEVAL: analyze question intent → Embedding → Hybrid Search 
 * (documents: whole vectorDB | transcripts: user-specific chunks) → 
 * rerank chunks → Verification Layer (boolean YES/NO) → 
 * send to LLM with user query to generate response
 */
export async function processQuery(query: string, userId?: string): Promise<{
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

        // Execute the complete retrieval flow
        processingSteps.push("🚀 Executing retrieval pipeline:");
        processingSteps.push("  Step 1: Analyze question intent");
        processingSteps.push("  Step 2: Embedding → Hybrid Search");
        processingSteps.push("  Step 3: Rerank chunks"); 
        processingSteps.push("  Step 4: Get parent context");
        processingSteps.push("  Step 5: Verification Layer (boolean YES/NO)");
        processingSteps.push("  Step 6: Generate response (if verified)");

        const retrievalResult = await retrieveContext(query);
        
        // Log retrieval success
        if (retrievalResult.chunks.length > 0) {
            processingSteps.push(`✅ Retrieval successful: ${retrievalResult.chunks.length} relevant chunks found`);
            processingSteps.push(`📚 Sources: ${retrievalResult.sources.join(', ')}`);
            processingSteps.push(`🎯 Strategy: ${retrievalResult.searchStrategy}`);
            processingSteps.push(`📊 Confidence: ${(retrievalResult.confidence * 100).toFixed(1)}%`);
        } else {
            processingSteps.push("❌ No relevant content found after verification");
        }

        // Generate response using LangChain
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
                chunks: [],
                sources: [],
                confidence: 0,
                searchStrategy: 'error'
            },
            processingSteps
        };
    }
}

/**
 * Simple query interface for basic usage
 */
export async function askQuestion(question: string, userId?: string): Promise<string> {
    const result = await processQuery(question, userId);
    return result.answer;
}

/**
 * Debug query interface that returns detailed processing information
 */
export async function debugQuery(question: string, userId?: string): Promise<{
    answer: string;
    debug: {
        retrievalResult: RetrievalResult;
        processingSteps: string[];
        metadata: {
            totalChunks: number;
            sources: string[];
            confidence: number;
            strategy: string;
        };
    };
}> {
    const result = await processQuery(question, userId);
    
    return {
        answer: result.answer,
        debug: {
            retrievalResult: result.retrievalResult,
            processingSteps: result.processingSteps,
            metadata: {
                totalChunks: result.retrievalResult.chunks.length,
                sources: result.retrievalResult.sources,
                confidence: result.retrievalResult.confidence,
                strategy: result.retrievalResult.searchStrategy
            }
        }
    };
}