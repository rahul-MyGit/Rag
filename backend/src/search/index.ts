import { CONFIG } from "../config";
import { type IntentAnalysis, type BM25Result, type HybridSearchResult } from "../types";
import { llm, docVectorStore, transcriptVectorStore, reranker } from "../config/initialize";
import { Document } from "langchain/document";
import { searchBM25 } from "../utils/bm25";

// Define SearchResult type
interface SearchResult {
    document: Document;
    score: number;
    source: "semantic" | "bm25";
}

export async function analyzeQueryIntent(query: string): Promise<IntentAnalysis> {
    const prompt = `Analyze the following query and provide a JSON response with the following fields:
  
  Query: "${query}"
  
  Analyze:
  1. needsDocuments: Does this query need information from formal documents? (boolean)
  2. needsTranscripts: Does this query need information from conversation transcripts? (boolean)
  3. userId: If transcript info is needed, which user? ("nathan", "robert", "both", or null)
  4. confidence: How confident are you in this analysis? (0.0-1.0)
  5. queryType: What type of query is this? ("factual", "conversational", "analytical", "mixed")
  
  Examples:
  - "What are the key features mentioned in the documentation?" → needsDocuments: true, needsTranscripts: false
  - "What did Nathan say about the project?" → needsDocuments: false, needsTranscripts: true, userId: "nathan"
  - "Compare the documentation with Robert's feedback" → needsDocuments: true, needsTranscripts: true, userId: "robert"
  
  Respond only with valid JSON:`;

    try {
        const response = await llm.invoke(prompt);
        return JSON.parse(response.content as string);
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
    topK: number = 10
): Promise<SearchResult[]> {
    if (!transcriptVectorStore) {
        throw new Error("Transcript vector store not initialized");
    }

    // Get all embeddings for this client first
    const clientFilter = {
        filter: {
            clientId: userId
        }
    };

    // Get all documents for this client
    const clientDocs = await transcriptVectorStore.similaritySearch("", 1000, clientFilter);

    // Perform semantic search on client docs
    const semanticResults = await transcriptVectorStore.similaritySearchWithScore(
        query,
        topK,
        clientFilter
    );

    // Perform BM25 search on client docs
    const bm25Results = await searchBM25(
        query,
        "transcripts",
        topK,
        clientDocs
    );

    // Combine and rerank results
    const combinedResults = new Map<string, SearchResult>();

    // Add semantic results
    semanticResults.forEach(([doc, score]: [Document, number]) => {
        combinedResults.set(doc.metadata.id, {
            document: doc,
            score: score,
            source: "semantic"
        });
    });

    // Add BM25 results
    bm25Results.forEach(({ document, score }) => {
        const existing = combinedResults.get(document.metadata.id);
        if (existing) {
            // If document exists, average the scores
            existing.score = (existing.score + score) / 2;
        } else {
            combinedResults.set(document.metadata.id, {
                document,
                score,
                source: "bm25"
            });
        }
    });

    // Convert to array and sort by score
    return Array.from(combinedResults.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
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

export async function rerankWithCohere(results: HybridSearchResult[], query: string): Promise<Document[]> {
    if (results.length === 0) return [];

    try {
        const documents = results.map(result => result.document);
        const rerankedResults = await reranker.rerank(documents, query, {
            topN: CONFIG.SEARCH.RERANK_TOP_K
        });

        // Map reranked results back to original documents and filter out any undefined
        return rerankedResults
            .map(result => documents[result.index])
            .filter((doc): doc is Document => doc !== undefined);
    } catch (error) {
        console.error('Error in Cohere reranking:', error);
        return results.slice(0, CONFIG.SEARCH.RERANK_TOP_K).map(result => result.document);
    }
}

export async function verifyRelevance(chunks: Document[], query: string): Promise<boolean> {
    if (chunks.length === 0) return false;

    const context = chunks.slice(0, 3).map(chunk =>
        chunk.pageContent.substring(0, 500)
    ).join('\n\n');

    const prompt = `You are a strict relevance verification system. Your job is to determine if the provided context contains relevant information to answer the user's question.
  
  Context:
  ${context}
  
  Question: ${query}
  
Evaluation criteria:
  - Does the context directly address the question?
  - Is there enough information to provide a meaningful answer?
  - Are the key concepts from the question present in the context?
- Can a reasonable answer be constructed from this context?

CRITICAL: You must respond with EXACTLY one word:
- "YES" if the context is sufficiently relevant to answer the question
- "NO" if the context is not relevant or insufficient
  
Response:`;

    try {
        const response = await llm.invoke(prompt);
        return (response.content as string).trim().toUpperCase() === 'YES';
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