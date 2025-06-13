import { CONFIG } from "../config";
import { type BM25Result, type HybridSearchResult, type QueryIntent } from "../types";
import { llm, docVectorStore, transcriptVectorStore } from "../config/initialize";
import { Document } from "langchain/document";
import { searchBM25 } from "../utils/bm25";

interface SearchResult {
    document: Document;
    score: number;
    source: "semantic" | "bm25";
}

export async function analyzeQueryIntent(query: string): Promise<QueryIntent> {
    const prompt = `Analyze the following query and determine if it needs document or transcript information or both.
Query: "${query}"

Consider the following guidelines:
- If query is about policy, guidelines, rules, compliance, or violations - it needs BOTH documents and transcripts (queryType: "mixed")
- If query is about specific conversations, interactions, or "what did I say" - it needs transcripts (queryType: "transcript")
- If query is about general information, documentation, or procedures - it needs documents (queryType: "document")
- If query combines conversation content with policy/guidelines - it needs both (queryType: "mixed")

Examples:
- "Did I say anything against policy on call?" → needs both documents (policy info) and transcripts (conversation content) → "mixed"
- "What medication did I suggest?" → needs transcripts only → "transcript"
- "What is the company policy on..." → needs documents only → "document"

Respond with a JSON object in this exact format:
{
    "needsDocuments": boolean,
    "needsTranscripts": boolean,
    "confidence": number between 0 and 1,
    "queryType": "document" | "transcript" | "mixed"
}

Respond only with valid JSON:`;

    try {
        const response = await llm.invoke(prompt);
        const content = response.content as string;
        const cleanedContent = content.replace(/```json\n?|\n?```/g, '').trim();
        return JSON.parse(cleanedContent);
    } catch (error) {
        console.error('Error analyzing query intent:', error);
        return {
            needsDocuments: true,
            needsTranscripts: true,
            confidence: 0.5,
            queryType: 'mixed'
        };
    }
}

export async function performHybridSearch(
    query: string,
    userId: number,
    type: 'documents' | 'transcripts' = 'transcripts',
    topK: number = 10
): Promise<SearchResult[]> {
    const vectorStore = type === 'documents' ? docVectorStore : transcriptVectorStore;
    
    if (!vectorStore) {
        throw new Error(`${type} vector store not initialized`);
    }

    console.log(`\n=== HYBRID SEARCH DEBUG ===`);
    console.log(`Query: "${query}"`);
    console.log(`Type: ${type}, UserId: ${userId}, TopK: ${topK}`);

    let clientFilter: Record<string, any> = {};
    let clientDocs: Document[] = [];

    if (type === 'transcripts') {
        clientFilter = { clientId: userId };
        clientDocs = await vectorStore.similaritySearch("", 1000, clientFilter);
        console.log(`Pre-filtered TRANSCRIPT documents for user ${userId}: ${clientDocs.length}`);
    } else {
        clientDocs = await vectorStore.similaritySearch("", 1000);
        console.log(`Pre-filtered DOCUMENT documents (global): ${clientDocs.length}`);
    }

    let semanticResults: [Document, number][];
    if (type === 'transcripts') {
        semanticResults = await vectorStore.similaritySearchWithScore(
            query,
            topK,
            clientFilter
        );
    } else {
        semanticResults = await vectorStore.similaritySearchWithScore(
            query,
            topK
        );
    }

    console.log(`Semantic search results: ${semanticResults.length}`);
    semanticResults.forEach(([doc, score], index) => {
        console.log(`  Semantic ${index + 1}: Score ${score.toFixed(4)}, Source: ${doc.metadata.source || doc.metadata.fileName}`);
        console.log(`    Content preview: "${doc.pageContent.substring(0, 150)}..."`);
    });

    const bm25Results = await searchBM25(
        query,
        type,
        topK,
        clientDocs
    );
    console.log(`BM25 search results: ${bm25Results.length}`);
    bm25Results.forEach(({ document, score }, index) => {
        console.log(`  BM25 ${index + 1}: Score ${score.toFixed(4)}, Source: ${document.metadata.source || document.metadata.fileName}`);
        console.log(`    Content preview: "${document.pageContent.substring(0, 150)}..."`);
    });

    const combinedResults = new Map<string, SearchResult>();

    semanticResults.forEach(([doc, score]: [Document, number]) => {
        const id = doc.metadata.source || doc.metadata.fileName || doc.metadata.id;
        combinedResults.set(id, {
            document: doc,
            score: score,
            source: "semantic"
        });
    });

    bm25Results.forEach(({ document, score }) => {
        const id = document.metadata.source || document.metadata.fileName || document.metadata.id;
        const existing = combinedResults.get(id);
        if (existing) {
            existing.score = (existing.score + score) / 2;
            console.log(`  Combined result: ${id}, Final score: ${existing.score.toFixed(4)}`);
        } else {
            combinedResults.set(id, {
                document,
                score,
                source: "bm25"
            });
        }
    });

    const finalResults = Array.from(combinedResults.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);

    console.log(`Final combined results: ${finalResults.length}`);
    finalResults.forEach((result, index) => {
        console.log(`  Final ${index + 1}: Score ${result.score.toFixed(4)}, Source: ${result.source}, File: ${result.document.metadata.source || result.document.metadata.fileName}`);
    });
    console.log(`=== END HYBRID SEARCH DEBUG ===\n`);

    return finalResults;
}

export function combineSearchResults(
    semanticResults: [Document, number][],
    bm25Results: BM25Result[],
    type: string
): HybridSearchResult[] {
    const semanticMap = new Map(semanticResults.map(([doc, score]) => [
        doc.metadata.source, 
        score / Math.max(...semanticResults.map(([, s]) => s))
    ]));

    const bm25Map = new Map(bm25Results.map(result => [
        result.document.metadata.source,
        result.score / Math.max(...bm25Results.map(r => r.score))
    ]));

    return Array.from(new Set([
        ...semanticResults.map(([doc]) => doc.metadata.source),
        ...bm25Results.map(result => result.document.metadata.source)
    ])).map(source => {
        const semanticScore = semanticMap.get(source) || 0;
        const bm25Score = bm25Map.get(source) || 0;
        const document = semanticResults.find(([doc]) => doc.metadata.source === source)?.[0] 
            || bm25Results.find(result => result.document.metadata.source === source)!.document;

        return {
            document,
            semanticScore,
            bm25Score,
            combinedScore: (semanticScore * CONFIG.SEARCH.SEMANTIC_WEIGHT) + 
                          (bm25Score * CONFIG.SEARCH.BM25_WEIGHT)
        };
    });
}

export async function verifyRelevance(chunks: Document[], query: string): Promise<boolean> {
    if (chunks.length === 0) return false;

    const context = chunks.slice(0, 3).map(chunk =>
        chunk.pageContent.substring(0, 500)
    ).join('\n\n');

    console.log(`\n=== VERIFICATION LAYER DEBUG ===`);
    console.log(`Query: "${query}"`);
    console.log(`Number of chunks: ${chunks.length}`);
    console.log(`Context preview (first 300 chars): "${context.substring(0, 300)}..."`);
    console.log(`Chunk sources: ${chunks.map(c => c.metadata.source || c.metadata.fileName).join(', ')}`);

    const prompt = `You are a relevance verification system. Your job is to determine if the provided context contains relevant information to answer the user's question.
  
  Context:
  ${context}
  
  Question: ${query}
  
Evaluation criteria:
  - Does the context directly address the question?
  - Is there enough information to provide a meaningful answer?
  - Are the key concepts from the question present in the context?
  - Can a reasonable answer be constructed from this context?
  - For health-related queries, even indirect mentions of health topics should be considered relevant

Be more lenient for conversational queries where partial information might still be useful.

CRITICAL: You must respond with EXACTLY one word:
- "YES" if the context is sufficiently relevant to answer the question
- "NO" if the context is not relevant or insufficient
  
Response:`;

    try {
        const response = await llm.invoke(prompt);
        const result = (response.content as string).trim().toUpperCase() === 'YES';
        console.log(`Verification result: ${result ? 'YES (RELEVANT)' : 'NO (NOT RELEVANT)'}`);
        console.log(`=== END VERIFICATION DEBUG ===\n`);
        return result;
    } catch (error) {
        console.error('Error in relevance verification:', error);
        return false;
    }
}

export async function reformulateQuery(originalQuery: string, retrievalAttempts: number = 1): Promise<string> {
    const prompt = `The following query didn't return relevant results. Please reformulate it to be more specific and searchable.
  
  Original query: "${originalQuery}"
  Attempt number: ${retrievalAttempts}
  
  Guidelines for reformulation:
  - Make it more specific and targeted
  - Use alternative keywords or synonyms
  - Break down complex queries into simpler parts
  - Focus on key concepts that are likely to appear in documents
  
  Reformulated query:`;

    try {
        const response = await llm.invoke(prompt);
        return (response.content as string).trim();
    } catch (error) {
        console.error('Error reformulating query:', error);
        return originalQuery;
    }
}