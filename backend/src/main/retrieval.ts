import { Document } from "langchain/document";
import { analyzeQueryIntent } from "../search";
import { type RetrievalResult } from "../types";
import { docRetriever, embeddings } from "../config/initialize";
import { ChatOpenAI } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { performPineconeHybridSearch, type HybridSearchResult } from "../utils/pinecone-hybrid-search";
import fs from 'fs';
import path from 'path';



const chatModel = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0.7,
});

function reciprocalRankFusion(results: HybridSearchResult[], k: number = 60): Document[] {
    const denseRanked = [...results].sort((a, b) => b.denseScore - a.denseScore);
    const sparseRanked = [...results].sort((a, b) => b.sparseScore - a.sparseScore);

    const rrfScores = new Map<string, number>();

    denseRanked.forEach((result, index) => {
        const docId = result.document.metadata.source || result.document.metadata.id;
        const score = 1 / (k + index + 1);
        rrfScores.set(docId, (rrfScores.get(docId) || 0) + score);
    });

    sparseRanked.forEach((result, index) => {
        const docId = result.document.metadata.source || result.document.metadata.id;
        const score = 1 / (k + index + 1);
        rrfScores.set(docId, (rrfScores.get(docId) || 0) + score);
    });

    return Array.from(rrfScores.entries())
        .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
        .slice(0, 10)
        .map(([docId]) => {
            return results.find(r =>
                (r.document.metadata.source || r.document.metadata.id) === docId
            )?.document;
        })
        .filter((doc): doc is Document => doc !== undefined);
}

export async function retrieveContext(query: string, userId: string): Promise<RetrievalResult> {
    console.log(`\n=== RETRIEVAL PROCESS ===`);
    console.log(`Query: "${query}"`);

        try {
            const [intent, queryEmbedding] = await Promise.all([
            analyzeQueryIntent(query),
            embeddings.embedQuery(query)
            ]);

            console.log('Step 1 - Intent analysis:', intent);
            console.log(`Step 2 - Query embedded successfully (${queryEmbedding.length} dimensions)`);

            const clientId = userId === 'nathan' ? 1 :
                userId === 'robert' ? 2 :
                    0; // Default to 0 for no specific user

            console.log(`Step 3 - Using clientId: ${clientId} for user: ${userId || 'none'}`);

        // Step 4: Execute agentic retrieval strategy based on intent
        let finalChunks: Document[] = [];

            if (intent.queryType === 'document') {
            finalChunks = await handleDocumentOnlyQuery(query);
            } else if (intent.queryType === 'transcript') {
            finalChunks = await handleTranscriptOnlyQuery(query, userId);
                } else {
            finalChunks = await handleMixedQuery(query, userId);
        }

        console.log(`Step 4 - Agentic retrieval complete: ${finalChunks.length} final chunks`);

        if (finalChunks.length === 0) {
                return {
                    content: '',
                    confidence: 0,
                    sources: [],
                    searchStrategy: 'failed'
                };
            }

                const sources = [...new Set(finalChunks.map(chunk => chunk.metadata.fileName))];
        console.log('Step 5 - Generating answer with retrieved context');

                const contextText = finalChunks.map(chunk => chunk.pageContent).join('\n\n');

        const prompt = `You are an expert assistant specializing in personalized recommendations and answering questions. Only based on below context answer the question. Don't add any extra line in starting or ending of the answer.

Context:
${contextText}

User Question: ${query}

CRITICAL INSTRUCTIONS: 
1. Answer the question based on the context provided only.
2. Don't add any extra line in starting or ending of the answer.
3. Answer should be very specific and to the point.

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
        console.error(`Error in retrieval:`, error);
        throw error;
    }
}

/**
 * Agentic function to handle document-only queries
 */
async function handleDocumentOnlyQuery(query: string): Promise<Document[]> {
    console.log('🔍 Executing DOCUMENT-ONLY search strategy with Pinecone Hybrid Search');
    
    const hybridResults = await performPineconeHybridSearch(query, 'documents', 0, 10);
    console.log(`📚 Found ${hybridResults.length} hybrid document results`);
    
    const rerankedDocuments = reciprocalRankFusion(hybridResults);
    console.log(`🔄 RRF reranked to ${rerankedDocuments.length} documents`);
    
    // Handle parent-child document retrieval
    let finalChunks: Document[] = [];
    
    for (const childDoc of rerankedDocuments) {
        if (childDoc.metadata.parentId && childDoc.metadata.type === 'document') {
            try {
                if (docRetriever) {
                    const parentResults = await docRetriever.invoke(childDoc.pageContent);
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
    
    console.log(`✅ Document-only strategy complete: ${finalChunks.length} final chunks`);
    return finalChunks;
}

/**
 * Agentic function to handle transcript-only queries
 */
async function handleTranscriptOnlyQuery(query: string, userId: string): Promise<Document[]> {
    console.log('🗣️ Executing TRANSCRIPT-ONLY search strategy with Pinecone Hybrid Search');
    
    const clientId = userId === 'nathan' ? 1 : userId === 'robert' ? 2 : 0;
    
    if (clientId === 0) {
        throw new Error('Invalid user for transcript search. Must be nathan or robert.');
    }
    
    const hybridResults = await performPineconeHybridSearch(query, 'transcripts', clientId, 10);
    console.log(`💬 Found ${hybridResults.length} hybrid transcript results for ${userId}`);
    
    const rerankedDocuments = reciprocalRankFusion(hybridResults);
    console.log(`🔄 RRF reranked to ${rerankedDocuments.length} transcript chunks`);
    
    console.log(`✅ Transcript-only strategy complete: ${rerankedDocuments.length} final chunks`);
    return rerankedDocuments;
}

/**
 * Helper function to get the latest PDF from user's transcript directory
 * Handles date-based naming like: nathan-03-14.pdf, robert-12-25.pdf
 */
async function getLatestTranscript(userName: string): Promise<string> {
    const userDir = path.resolve(`D:\\100xSuper30\\rag-playground\\transcriptions_for_test\\${userName}`);
    
    if (!fs.existsSync(userDir)) {
        throw new Error(`User directory not found: ${userDir}`);
    }
    
    const files = fs.readdirSync(userDir)
        .filter(file => file.toLowerCase().endsWith('.pdf'))
        .map(file => {
                        // Parse date from filename: {name}-{month}-{day}.pdf (handles both single and double digit months/days)
            const dateMatch = file.match(/(\d{1,2})-(\d{1,2})\.pdf$/);
            let sortKey = file; // fallback to filename
            
            if (dateMatch) {
                const month = parseInt(dateMatch[1] || '0');
                const day = parseInt(dateMatch[2] || '0');
                sortKey = `${month.toString().padStart(2, '0')}${day.toString().padStart(2, '0')}`;
            }

    return {
                name: file,
                path: path.join(userDir, file),
                sortKey,
                stats: fs.statSync(path.join(userDir, file))
            };
        })
        .sort((a, b) => {
            if (a.sortKey !== a.name && b.sortKey !== b.name) {
                return b.sortKey.localeCompare(a.sortKey);
            }
            return b.stats.mtime.getTime() - a.stats.mtime.getTime();
        });
    
    if (files.length === 0) {
        throw new Error(`No PDF files found in ${userDir}`);
    }
    
    const latestFile = files[0];
    console.log(`📄 Loading latest transcript: ${latestFile?.name} (sort key: ${latestFile?.sortKey})`);
    
    const loader = new PDFLoader(latestFile?.path || '');
    const docs = await loader.load();
    const fullText = docs.map(doc => doc.pageContent).join('\n\n');
    
    console.log(`✅ Loaded ${fullText.length} characters from latest transcript`);
    return fullText;
}

/**
 * Agentic function to handle mixed queries (documents + latest transcript)
 */
async function handleMixedQuery(query: string, userId: string): Promise<Document[]> {
    console.log('🔄 Executing MIXED search strategy (documents + latest transcript)');
    
    // Step 1: Get document results
    console.log('Step 1: Searching documents...');
    const documentResults = await handleDocumentOnlyQuery(query);
    console.log(`📚 Retrieved ${documentResults.length} document chunks`);
    
    // Step 2: Get latest transcript text (no search, just parse)
    let userName = '';
    let clientId = 0;
    
    if (userId === 'nathan' || userId === '1') {
        userName = 'nathan';
        clientId = 1;
    } else if (userId === 'robert' || userId === '2') {
        userName = 'robert';
        clientId = 2;
    } else {
        userName = 'unknown';
        clientId = 0;
    }
    
    console.log('Step 2: Loading latest transcript...');
    try {
        const latestTranscriptText = await getLatestTranscript(userName);
        
        // Create a document object for the latest transcript
        const transcriptDocument = new Document({
            pageContent: latestTranscriptText,
            metadata: {
                source: 'latest_transcript',
                fileName: `${userName}_latest_transcript`,
                type: 'transcript',
                userId: userName,
                clientId: clientId
            }
        });
        
        console.log(`💬 Added latest transcript (${latestTranscriptText.length} characters)`);
        
        // Combine document results with latest transcript
        const combinedResults = [...documentResults, transcriptDocument];
        
        console.log(`✅ Mixed strategy complete: ${documentResults.length} documents + 1 latest transcript = ${combinedResults.length} total chunks`);
        return combinedResults;
        
    } catch (error) {
        console.warn(`⚠️ Failed to load latest transcript for ${userId}: ${error}`);
        console.log('📚 Falling back to documents only');
        return documentResults;
    }
}
