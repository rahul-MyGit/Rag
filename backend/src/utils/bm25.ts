import { Document } from "langchain/document";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import { bm25Storage } from "../config/initialize";
import { type BM25Result } from "../types";

export function createBM25Index(documents: Document[], indexName: string): void {
    const bm25Retriever = BM25Retriever.fromDocuments(documents, { k: 10 });

    bm25Storage[indexName as keyof typeof bm25Storage].set('index', bm25Retriever);
    bm25Storage[indexName as keyof typeof bm25Storage].set('documents', documents);
}

function sanitizeQueryForBM25(query: string): string {
    return query
        .replace(/[.*+?^${}()|[\]\\]/g, ' ')
        .replace(/['"]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}

export async function searchBM25(
    query: string, 
    indexName: string, 
    topK: number = 10,
    preFilteredDocs?: Document[]
): Promise<BM25Result[]> {
    const storage = bm25Storage[indexName as keyof typeof bm25Storage];
    const bm25Retriever = storage.get('index') as BM25Retriever;

    if (!bm25Retriever) {
        return [];
    }

    try {
        const sanitizedQuery = sanitizeQueryForBM25(query);
        console.log(`BM25 Query: "${query}" → Sanitized: "${sanitizedQuery}"`);

        bm25Retriever.k = topK;

        if (preFilteredDocs && preFilteredDocs.length > 0) {
            const tempRetriever = BM25Retriever.fromDocuments(preFilteredDocs, { k: topK });
            const retrievedDocs = await tempRetriever.invoke(sanitizedQuery);
            return retrievedDocs.map((doc, index) => ({
                document: doc,
                score: 1 - (index * 0.1)
            }));
        }

        const retrievedDocs = await bm25Retriever.invoke(sanitizedQuery);
        return retrievedDocs.map((doc, index) => ({
            document: doc,
            score: 1 - (index * 0.1)
        }));
    } catch (error) {
        console.error('Error in BM25 search:', error);
        console.error('Original query:', query);
        return [];
    }
}