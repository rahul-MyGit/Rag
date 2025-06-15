import { Document } from "langchain/document";
import { analyzeQueryIntent, performHybridSearch, reformulateQuery } from "../search";
import { type RetrievalResult, type HybridSearchResult } from "../types";
import { docRetriever, embeddings } from "../config/initialize";
import { ChatOpenAI } from "@langchain/openai";

// Optimized model for verification - using GPT-4o-mini for speed + accuracy
const verificationModel = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0.1, // Lower temperature for more consistent verification
});

// Main model for final answer generation - using GPT-4o-mini for speed + quality
const chatModel = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0.7,
});

function reciprocalRankFusion(results: HybridSearchResult[], k: number = 60): Document[] {
    const semanticRanked = [...results].sort((a, b) => b.semanticScore - a.semanticScore);

    const bm25Ranked = [...results].sort((a, b) => b.bm25Score - a.bm25Score);

    const rrfScores = new Map<string, number>();

    semanticRanked.forEach((result, index) => {
        const docId = result.document.metadata.source || result.document.metadata.id;
        const score = 1 / (k + index + 1);
        rrfScores.set(docId, (rrfScores.get(docId) || 0) + score);
    });

    bm25Ranked.forEach((result, index) => {
        const docId = result.document.metadata.source || result.document.metadata.id;
        const score = 1 / (k + index + 1);
        rrfScores.set(docId, (rrfScores.get(docId) || 0) + score);
    });

    return Array.from(rrfScores.entries())
        .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
        .slice(0, 5)
        .map(([docId]) => {
            return results.find(r =>
                (r.document.metadata.source || r.document.metadata.id) === docId
            )?.document;
        })
        .filter((doc): doc is Document => doc !== undefined);
}

export async function retrieveContext(query: string, userId?: string, maxAttempts: number = 3): Promise<RetrievalResult> {
    let currentQuery = query;
    let attempts = 0;

    while (attempts < maxAttempts) {
        attempts++;
        console.log(`\n=== RETRIEVAL ATTEMPT ${attempts}/${maxAttempts} ===`);
        console.log(`Query: "${currentQuery}"`);
        if (attempts > 1) {
            console.log(`🔄 This is a reformulated query from attempt ${attempts - 1}`);
        }

        try {
            // Step 1: Analyze intent and embed query in parallel
            const [intent, queryEmbedding] = await Promise.all([
                analyzeQueryIntent(currentQuery),
                embeddings.embedQuery(currentQuery)
            ]);
            
            console.log('Step 1 - Intent analysis:', intent);
            console.log(`Step 2 - Query embedded successfully (${queryEmbedding.length} dimensions)`);
            if (attempts > 1) {
                console.log(`✅ Reformulated query has been re-embedded for fresh search`);
            }

            const clientId = userId === 'nathan' ? 1 :
                userId === 'robert' ? 2 :
                    0; // Default to 0 for no specific user

            console.log(`Step 3 - Using clientId: ${clientId} for user: ${userId || 'none'}`);


            let searchResults: any[] = [];

            if (intent.queryType === 'document') {
                searchResults = await performHybridSearch(currentQuery, 0, 'documents');
            } else if (intent.queryType === 'transcript') {
                if (!userId) {
                    return {
                        content: '',
                        confidence: 0,
                        sources: [],
                        searchStrategy: 'failed'
                    };
                }
                searchResults = await performHybridSearch(currentQuery, clientId, 'transcripts');
            } else if (intent.queryType === 'document_then_transcript') {
                console.log('🔄 Initiating DOCUMENT-FIRST cross-reference retrieval...');
                
                // Step 1: Search documents to understand the policy/principle/intervention
                console.log('Step 1: Searching documents for policy/principle definition...');
                const docResults = await performHybridSearch(currentQuery, 0, 'documents');
                
                if (docResults.length > 0 && userId) {
                    // Step 2: Extract key concepts from document results to enhance transcript search
                    const policyContext = docResults.slice(0, 3).map(r => r.document.pageContent).join(' ');
                    console.log(`📋 Policy context extracted (${policyContext.length} chars)`);
                    
                    // Step 3: Search transcripts for evidence of policy application
                    console.log('Step 2: Searching transcripts for evidence of policy application...');
                    const enhancedTranscriptQuery = `${currentQuery} evidence of: ${policyContext.substring(0, 500)}`;
                    const transcriptResults = await performHybridSearch(enhancedTranscriptQuery, clientId, 'transcripts');
                    
                    // Combine results: documents first (for context), then transcript evidence
                    searchResults = [...docResults.slice(0, 2), ...transcriptResults];
                    console.log(`✅ Document-first cross-reference complete: ${docResults.length} policy docs + ${transcriptResults.length} transcript evidence`);
                } else {
                    // Fallback to document-only if no user context or no document results
                    console.log('❌ No policy context found or no user specified, using document results only');
                    searchResults = docResults;
                }
            } else if (intent.queryType === 'transcript_then_document') {
                console.log('🔄 Initiating TRANSCRIPT-FIRST cross-reference retrieval...');
                
                if (!userId) {
                    return {
                        content: '',
                        confidence: 0,
                        sources: [],
                        searchStrategy: 'failed'
                    };
                }
                
                // Step 1: Search transcripts to understand client situation/context
                console.log('Step 1: Searching transcripts for client context...');
                const transcriptResults = await performHybridSearch(currentQuery, clientId, 'transcripts');
                
                if (transcriptResults.length > 0) {
                    // Step 2: Extract client context to enhance document search
                    const clientContext = transcriptResults.slice(0, 3).map(r => r.document.pageContent).join(' ');
                    console.log(`📋 Client context extracted (${clientContext.length} chars)`);
                    
                    // Step 3: Search documents for relevant policies/interventions
                    console.log('Step 2: Searching documents for relevant policies/interventions...');
                    const enhancedDocQuery = `${currentQuery} for client situation: ${clientContext.substring(0, 500)}`;
                    const docResults = await performHybridSearch(enhancedDocQuery, 0, 'documents');
                    
                    // Combine results: transcript context first, then relevant policies
                    searchResults = [...transcriptResults.slice(0, 2), ...docResults];
                    console.log(`✅ Transcript-first cross-reference complete: ${transcriptResults.length} client context + ${docResults.length} policy recommendations`);
                } else {
                    console.log('❌ No client context found, cannot perform cross-reference');
                    return {
                        content: '',
                        confidence: 0,
                        sources: [],
                        searchStrategy: 'failed'
                    };
                }
            } else {
                // Cross-index retrieval: First get user context, then search documents
                if (userId && (currentQuery.toLowerCase().includes('recommend') || 
                               currentQuery.toLowerCase().includes('suggest') || 
                               currentQuery.toLowerCase().includes('program'))) {
                    console.log('🔄 Initiating cross-index retrieval...');
                    
                    // Step 1: Get user context from transcripts
                    console.log('Step 1: Retrieving user context from transcripts...');
                    const userContextQuery = `${userId} client background needs goals progress treatment status situation`;
                    const userContextResults = await performHybridSearch(userContextQuery, clientId, 'transcripts');
                    
                    if (userContextResults.length > 0) {
                        // Extract user context for enhanced document search
                        const userContext = userContextResults.slice(0, 3).map(r => r.document.pageContent).join(' ');
                        console.log(`📋 User context extracted (${userContext.length} chars)`);
                        
                        // Step 2: Use user context to enhance document search
                        console.log('Step 2: Searching documents with user context...');
                        const enhancedQuery = `${currentQuery} for client with: ${userContext.substring(0, 500)}`;
                        const docResults = await performHybridSearch(enhancedQuery, 0, 'documents');
                        
                        // Combine results with user context first (for context) and relevant docs
                        searchResults = [...userContextResults.slice(0, 2), ...docResults];
                        console.log(`✅ Cross-index retrieval complete: ${userContextResults.length} user context + ${docResults.length} document results`);
                    } else {
                        // Fallback to regular mixed search
                        console.log('❌ No user context found, falling back to standard search');
                        const [docResults, transcriptResults] = await Promise.all([
                            performHybridSearch(currentQuery, 0, 'documents'),
                            performHybridSearch(currentQuery, clientId, 'transcripts')
                        ]);
                        searchResults = [...docResults, ...transcriptResults];
                    }
                } else {
                    // Regular mixed search for non-recommendation queries
                    const [docResults, transcriptResults] = await Promise.all([
                        performHybridSearch(currentQuery, 0, 'documents'),
                        userId ? performHybridSearch(currentQuery, clientId, 'transcripts') : Promise.resolve([])
                    ]);
                    searchResults = [...docResults, ...transcriptResults];
                }
            }

            const hybridResults: HybridSearchResult[] = searchResults.map(result => ({
                document: result.document,
                semanticScore: result.source === 'semantic' ? result.score : 0,
                bm25Score: result.source === 'bm25' ? result.score : 0,
                combinedScore: result.score
            }));

            console.log(`Step 4 - Found ${hybridResults.length} hybrid search results`);

            if (hybridResults.length === 0) {
                if (attempts < maxAttempts) {
                    console.log('No results found, reformulating query...');
                    currentQuery = await reformulateQuery(currentQuery, attempts);
                    continue;
                }
                return {
                    content: '',
                    confidence: 0,
                    sources: [],
                    searchStrategy: 'failed'
                };
            }

            const rerankedDocuments = reciprocalRankFusion(hybridResults);
            console.log(`Step 5 - RRF reranked to ${rerankedDocuments.length} documents`);

            let finalChunks: Document[] = [];

            for (const childDoc of rerankedDocuments) {
                if (childDoc.metadata.parentId && childDoc.metadata.type === 'document') {
                    try {
                        if (docRetriever) {
                            const parentResults = await docRetriever.getRelevantDocuments(childDoc.pageContent);
                            if (parentResults.length > 0 && parentResults[0]) {
                                finalChunks.push(parentResults[0] as Document);
                            } else {
                                finalChunks.push(childDoc);
                            }
                        } else {
                            finalChunks.push(childDoc);
                        }
                    } catch (error) {
                        console.warn('Failed to get parent chunk, using child:', error);
                        finalChunks.push(childDoc);
                    }
                } else {
                    finalChunks.push(childDoc);
                }
            }

            console.log(`Step 6 - Retrieved ${finalChunks.length} final chunks with context`);

            const isRelevant = await verifyRelevanceOptimized(finalChunks, currentQuery);
            console.log(`Step 7 - Verification Layer: ${isRelevant ? 'YES (RELEVANT)' : 'NO (NOT RELEVANT)'}`);

            if (isRelevant) {
                console.log('✅ Verification passed - proceeding to answer generation');
                
                const sources = [...new Set(finalChunks.map(chunk => chunk.metadata.fileName))];
                console.log('Step 8 - Generating precise answer with verified relevant context');

                const contextText = finalChunks.map(chunk => chunk.pageContent).join('\n\n');

                const prompt = `You are an expert RAG assistant specializing in personalized recommendations. The context below contains VERIFIED RELEVANT information including client background and relevant guidance documents.

Context:
${contextText}

User Question: ${currentQuery}

CRITICAL INSTRUCTIONS: 
1. Answer the question based on the context provided only.
2. Don't add any extra line in starting or ending of the answer.
3 ans should be very specific and to the point.

Answer:`;

                const response = await chatModel.invoke(prompt);
                const generatedAnswer = typeof response.content === 'string'
                    ? response.content
                    : JSON.stringify(response.content);

                return {
                    content: generatedAnswer,
                    sources: sources.map(source => ({
                        type: 'document',
                        id: source,
                        score: 1.0
                    })),
                    confidence: intent.confidence,
                    searchStrategy: 'hybrid'
                };
            }

            if (!isRelevant && attempts < maxAttempts) {
                console.log('❌ LLM Verification: Content NOT relevant to query');
                console.log(`🔄 Reformulating query (attempt ${attempts}/${maxAttempts})...`);
                currentQuery = await reformulateQuery(currentQuery, attempts);
                console.log(`📝 New reformulated query: "${currentQuery}"`);
                console.log(`⚡ Will re-embed and search again with reformulated query`);
                continue;
            }

            if (!isRelevant && attempts === maxAttempts) {
                console.log('Max attempts reached with non-relevant content');
                return {
                    content: '',
                    confidence: 0,
                    sources: [],
                    searchStrategy: 'max-attempts-non-relevant'
                };
            }

        } catch (error) {
            console.error(`Error in retrieval attempt ${attempts}:`, error);
            if (attempts === maxAttempts) {
                throw error;
            }
        }
    }

    return {
        content: '',
        confidence: 0,
        sources: [],
        searchStrategy: 'max-attempts-reached'
    };
}

async function verifyRelevanceOptimized(chunks: Document[], query: string): Promise<boolean> {
    if (chunks.length === 0) return false;
    
    const context = chunks.map(chunk => chunk.pageContent).slice(0, 3).join('\n\n'); // Only check first 3 chunks for speed
    const sources = chunks.map(chunk => chunk.metadata.source || chunk.metadata.fileName).join(', ');
    
    console.log(`\n=== VERIFICATION LAYER DEBUG ===`);
    console.log(`Query: "${query}"`);
    console.log(`Number of chunks: ${chunks.length}`);
    console.log(`Context preview (first 300 chars): "${context.substring(0, 300)}..."`);
    console.log(`Chunk sources: ${sources}`);

    const verificationPrompt = `Analyze if the provided context is relevant to answer the user's question.

User Question: "${query}"

Context:
${context}

Instructions:
- Return ONLY "YES" if the context contains information that can help answer the question
- Return ONLY "NO" if the context is completely unrelated to the question
- Consider partial relevance as YES

Response (YES or NO only):`;

    try {
        const response = await verificationModel.invoke(verificationPrompt);
        const result = (response.content as string).trim().toUpperCase();
        const isRelevant = result.includes('YES');
        
        console.log(`Verification result: ${isRelevant ? 'YES (RELEVANT)' : 'NO (NOT RELEVANT)'}`);
        console.log(`=== END VERIFICATION DEBUG ===\n`);
        
        return isRelevant;
    } catch (error) {
        console.error('Error in verification:', error);
        console.log(`Verification result: YES (DEFAULT - due to error)`);
        console.log(`=== END VERIFICATION DEBUG ===\n`);
        return true;
    }
}

// async function retrieveFromPinecone(query: string, userId?: string): Promise<RetrievalResult> {
//     try {
//         const filter: Record<string, any> = {};
//         
//         // Only add userId filter if it's a valid string
//         if (userId && typeof userId === 'string' && userId !== 'both') {
//             filter.userId = userId;
//         }

//         const results = await pineconeIndex.query({
//             vector: await getEmbedding(query),
//             topK: 5,
//             filter,
//             includeMetadata: true
//         });

//         if (!results.matches || results.matches.length === 0) {
//             return {
//                 content: '',
//                 confidence: 0,
//                 sources: [],
//                 searchStrategy: 'semantic'
//             };
//         }

//         return {
//             content: results.matches.map((match: PineconeMatch) => match.metadata?.text || '').join('\n'),
//             confidence: results.matches[0].score || 0,
//             sources: results.matches.map((match: PineconeMatch) => ({
//                 type: match.metadata?.type || 'unknown',
//                 id: match.metadata?.id || 'unknown',
//                 score: match.score || 0
//             })),
//             searchStrategy: 'semantic'
//         };
//     } catch (error) {
//         console.error('Error retrieving from Pinecone:', error);
//         return {
//             content: '',
//             confidence: 0,
//             sources: [],
//             searchStrategy: 'semantic'
//         };
//     }
// }