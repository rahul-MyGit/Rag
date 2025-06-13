import { Document } from "langchain/document";
import { analyzeQueryIntent, performHybridSearch, verifyRelevance, reformulateQuery } from "../search";
import { type RetrievalResult, type HybridSearchResult } from "../types";
import { docRetriever, embeddings } from "../config/initialize";
import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
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
            const intent = await analyzeQueryIntent(currentQuery);
            console.log('Step 1 - Intent analysis:', intent);

            console.log('Step 2 - Embedding query...');
            const queryEmbedding = await embeddings.embedQuery(currentQuery);
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
            } else {
                const docResults = await performHybridSearch(currentQuery, 0, 'documents');
                const transcriptResults = userId ?
                    await performHybridSearch(currentQuery, clientId, 'transcripts') :
                    [];
                searchResults = [...docResults, ...transcriptResults];
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

            const isRelevant = await verifyRelevance(finalChunks, currentQuery);
            console.log(`Step 7 - Verification Layer: ${isRelevant ? 'YES (RELEVANT)' : 'NO (NOT RELEVANT)'}`);

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

            const sources = [...new Set(finalChunks.map(chunk => chunk.metadata.fileName))];
            console.log('Step 8 - Generating precise answer with verified relevant context');

            const contextText = finalChunks.map(chunk => chunk.pageContent).join('\n\n');

            const prompt = `You are a top-tier precise assistant. Based on the provided context, answer the user's question accurately and concisely only from context.

Context:
${contextText}

User Question: ${currentQuery}

Instructions:
- Provide a direct and precise answer based only on the information in the context
- If the context doesn't contain enough information to fully answer the question, clearly state what information is missing
- Use specific details and examples from the context when relevant
- If information comes from transcripts, you can mention the speaker when relevant
- Be concise
- Maintain a professional and helpful tone

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