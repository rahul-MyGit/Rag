import { Document } from "langchain/document";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import { bm25Storage } from "../config/initialize";
import { type BM25Result } from "../types";

export function createBM25Index(documents: Document[], indexName: string): void {
    // Create BM25 retriever with documents
    const bm25Retriever = BM25Retriever.fromDocuments(documents, { k: 10 });

    // Store retriever and documents
    bm25Storage[indexName as keyof typeof bm25Storage].set('index', bm25Retriever);
    bm25Storage[indexName as keyof typeof bm25Storage].set('documents', documents);
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
        // Update k parameter for retrieval count
        bm25Retriever.k = topK;

        // If we have pre-filtered documents, create a temporary retriever
        if (preFilteredDocs && preFilteredDocs.length > 0) {
            const tempRetriever = BM25Retriever.fromDocuments(preFilteredDocs, { k: topK });
            const retrievedDocs = await tempRetriever.invoke(query);
            return retrievedDocs.map((doc, index) => ({
                document: doc,
                score: 1 - (index * 0.1)
            }));
        }

        // Otherwise use the main retriever
        const retrievedDocs = await bm25Retriever.invoke(query);
        return retrievedDocs.map((doc, index) => ({
            document: doc,
            score: 1 - (index * 0.1)
        }));
    } catch (error) {
        console.error('Error in BM25 search:', error);
        return [];
    }
}