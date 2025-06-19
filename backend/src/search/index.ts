import { CONFIG } from "../config";
import { type BM25Result, type QueryIntent } from "../types";
import { llm, docDenseVectorStore, transcriptDenseVectorStore } from "../config/initialize";
import { Document } from "langchain/document";
import { searchBM25 } from "../utils/bm25";

interface SearchResult {
    document: Document;
    score: number;
    source: "semantic" | "bm25";
}

export async function analyzeQueryIntent(query: string): Promise<QueryIntent> {
    const prompt = `Classify this criminal probation query for search routing.

Query: "${query}"

Types:
- "document": Policy/procedure/law definitions, programs, and guidelines ("What is X?")
- "transcript": Client behavior/statements only ("What did client say?")  
- "mixed": Everything else including policy application and general queries

Decision rules:
- Only client behavior → transcript
- Only policy, procedure, definitions, programs, and guidelines → document
- Everything else → mixed

You must respond with ONLY a valid JSON object in this exact format:
{"needsDocuments":true,"needsTranscripts":true,"confidence":0.8,"queryType":"mixed"}

Do not include any explanation or additional text. Only the JSON object.`;

    try {
        const response = await llm.invoke(prompt);
        const content = response.content as string;
        
        let cleanedContent = content.replace(/```json\n?|\n?```/g, '').trim();
        
        const jsonMatch = cleanedContent.match(/\{[^}]*\}/);
        if (jsonMatch) {
            cleanedContent = jsonMatch[0];
        }
        
        const startIndex = cleanedContent.indexOf('{');
        const endIndex = cleanedContent.lastIndexOf('}');
        
        if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
            cleanedContent = cleanedContent.substring(startIndex, endIndex + 1);
        }
        
        console.log(`Raw LLM response: "${content}"`);
        console.log(`Cleaned JSON: "${cleanedContent}"`);
        
        return JSON.parse(cleanedContent);
    } catch (error) {
        console.error('Error analyzing query intent:', error);
        console.log('Falling back to default intent analysis');
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
    userId?: number,
    type: 'documents' | 'transcripts' = 'transcripts',
    topK: number = 10
): Promise<SearchResult[]> {
    const vectorStore = type === 'documents' ? docDenseVectorStore : transcriptDenseVectorStore;

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
        clientDocs = await vectorStore.similaritySearch("", Math.min(topK * 10, 500), clientFilter);
        console.log(`Pre-filtered TRANSCRIPT documents for user ${userId}: ${clientDocs.length}`);
    } else {
        clientDocs = await vectorStore.similaritySearch("", Math.min(topK * 10, 500));
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





