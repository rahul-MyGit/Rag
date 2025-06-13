import { Document } from "langchain/document";
import { analyzeQueryIntent, performHybridSearch, verifyRelevance, reformulateQuery, rerankWithCohere } from "../search";
import { type RetrievalResult, type HybridSearchResult } from "../types";
import { reranker, docRetriever, transcriptRetriever } from "../config/initialize";

export async function retrieveContext(query: string, maxAttempts: number = 3): Promise<RetrievalResult> {
    let currentQuery = query;
    let attempts = 0;

    while (attempts < maxAttempts) {
        attempts++;
        console.log(`Retrieval attempt ${attempts} for query: "${currentQuery}"`);

        try {
            // Step 1: Analyze question intent 
            const intent = await analyzeQueryIntent(currentQuery);
            console.log('Step 1 - Intent analysis:', intent);

            // Step 2: Embedding → Hybrid Search 
            const userId = intent.userId === 'nathan' ? 1 : intent.userId === 'robert' ? 2 : 0;
            const searchResults = await performHybridSearch(currentQuery, userId);
            
            // Convert SearchResult to HybridSearchResult
            const hybridResults: HybridSearchResult[] = searchResults.map(result => ({
                document: result.document,
                semanticScore: result.source === 'semantic' ? result.score : 0,
                bm25Score: result.source === 'bm25' ? result.score : 0,
                combinedScore: result.score
            }));

            console.log(`Step 2 - Found ${hybridResults.length} hybrid search results`);

            if (hybridResults.length === 0) {
                if (attempts < maxAttempts) {
                    console.log('No results found, reformulating query...');
                    currentQuery = await reformulateQuery(currentQuery, attempts);
                    continue;
                }
                return {
                    chunks: [],
                    sources: [],
                    confidence: 0,
                    searchStrategy: 'failed'
                };
            }

            // Step 3: Rerank chunks using Cohere
            const rerankedDocuments = await rerankWithCohere(hybridResults, currentQuery);
            console.log(`Step 3 - Reranked to ${rerankedDocuments.length} documents`);

            // Step 4: Get parent chunks for full context (from child chunks)
            let finalChunks: Document[] = [];
            
            for (const childDoc of rerankedDocuments) {
                // Get parent document using ParentDocumentRetriever if metadata has parentId
                if (childDoc.metadata.parentId) {
                    try {
                        // Try to get parent from document store
                        if (childDoc.metadata.type === 'document' && docRetriever) {
                            // For documents, get parent context
                            const parentResults = await docRetriever.getRelevantDocuments(childDoc.pageContent);
                            if (parentResults.length > 0 && parentResults[0]) {
                                finalChunks.push(parentResults[0] as Document);
                            } else {
                                finalChunks.push(childDoc);
                            }
                        } else if (childDoc.metadata.type === 'transcript' && transcriptRetriever) {
                            // For transcripts, get parent context
                            const parentResults = await transcriptRetriever.getRelevantDocuments(childDoc.pageContent);
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
                    // Already parent chunk or no parent relationship
                    finalChunks.push(childDoc);
                }
            }

            console.log(`Step 4 - Retrieved ${finalChunks.length} final chunks with context`);

            // Step 5: Verification Layer (strict boolean YES/NO check)
            const isRelevant = await verifyRelevance(finalChunks, currentQuery);
            console.log(`Step 5 - Verification Layer: ${isRelevant ? 'YES (RELEVANT)' : 'NO (NOT RELEVANT)'}`);

            if (!isRelevant && attempts < maxAttempts) {
                console.log('LLM returned NO - content not relevant, reformulating query...');
                currentQuery = await reformulateQuery(currentQuery, attempts);
                continue;
            }

            if (!isRelevant && attempts === maxAttempts) {
                console.log('Max attempts reached with non-relevant content');
                return {
                    chunks: [],
                    sources: [],
                    confidence: 0,
                    searchStrategy: 'max-attempts-non-relevant'
                };
            }

            // Step 6: Only proceed if LLM returned YES - send to LLM with user query
            const sources = [...new Set(finalChunks.map(chunk => chunk.metadata.fileName))];
            console.log('Step 6 - Proceeding to response generation with verified relevant context');

            return {
                chunks: finalChunks,
                sources,
                confidence: intent.confidence,
                searchStrategy: `verified-hybrid-retrieval-attempt-${attempts}`
            };

        } catch (error) {
            console.error(`Error in retrieval attempt ${attempts}:`, error);
            if (attempts === maxAttempts) {
                throw error;
            }
        }
    }

    return {
        chunks: [],
        sources: [],
        confidence: 0,
        searchStrategy: 'max-attempts-reached'
    };
}

async function getParentChunks(childChunks: Document[]): Promise<Document[]> {
    // In a real implementation, you'd fetch full parent chunks from storage
    // For now, we'll return the child chunks as they contain the parent context
    return childChunks;
}