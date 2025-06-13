import { docRetriever, transcriptRetriever } from "../config/initialize";
import { Document } from "langchain/document";
import { parsePDF, generateSummary, extractKeywords } from "../utils";
import { createBM25Index } from "../utils/bm25";
import fs from 'fs';
import path from 'path';

export async function ingestDocuments(): Promise<void> {
    if (!docRetriever) {
        throw new Error('Document retriever not initialized');
    }

    const docsPath = './docs_for_test';
    if (!fs.existsSync(docsPath)) {
        console.error('Documents directory not found:', docsPath);
        return;
    }

    const files = fs.readdirSync(docsPath).filter(file => file.endsWith('.pdf'));
    console.log(`Found ${files.length} document files to process`);

    const allParentDocuments: Document[] = [];

    for (const file of files) {
        console.log(`Processing document: ${file}`);
        const filePath = path.join(docsPath, file);
        const text = await parsePDF(filePath);
        
        // Generate summary and keywords for the document
        const summary = await generateSummary(text);
        const keywords = extractKeywords(text);

        // Create parent document with enhanced metadata
        const parentDoc = new Document({
            pageContent: text,
            metadata: {
                source: file,
                type: 'document',
                fileName: file,
                summary,
                keywords,
                timestamp: new Date().toISOString()
            }
        });

        allParentDocuments.push(parentDoc);
        console.log(`Created parent document from ${file}`);
    }

    // Add documents to ParentDocumentRetriever
    // This automatically handles:
    // 1. Parent splitting (1000 chars, 200 overlap)
    // 2. Child splitting (250 chars, 50 overlap) 
    // 3. Child embeddings → VectorDB
    // 4. Parent storage → DocStore
    await docRetriever.addDocuments(allParentDocuments);

    // Create BM25 index from parent documents for hybrid search
    createBM25Index(allParentDocuments, 'documents');

    console.log(`Successfully ingested ${allParentDocuments.length} parent documents with automatic child chunking`);
}

export async function ingestTranscripts(): Promise<void> {
    if (!transcriptRetriever) {
        throw new Error('Transcript retriever not initialized');
    }

    const transPath = './transcriptions_for_test';
    const files = fs.readdirSync(transPath).filter(file => file.endsWith('.pdf'));
    console.log(`Found ${files.length} transcript files to process`);

    const allParentDocuments: Document[] = [];

    for (const file of files) {
        console.log(`Processing transcript: ${file}`);
        const filePath = path.join(transPath, file);
        const text = await parsePDF(filePath);
        
        // Determine clientId from filename (nathan=1, robert=2)
        const clientId = file.toLowerCase().includes('nathan') ? 1 : 
                        file.toLowerCase().includes('robert') ? 2 : 0;
        
        const userName = clientId === 1 ? 'nathan' : 
                        clientId === 2 ? 'robert' : 'unknown';

        // Generate summary and keywords
        const summary = await generateSummary(text);
        const keywords = extractKeywords(text);

        // Extract timestamp from filename (format: name-MM-DD)
        const timestampMatch = file.match(/(\d{2}-\d{2})/);
        const timestamp = timestampMatch ? timestampMatch[1] : undefined;

        // Create parent document with clientId approach
        const parentDoc = new Document({
            pageContent: text,
            metadata: {
                source: file,
                type: 'transcript',
                clientId,
                userId: userName,
                fileName: file,
                summary,
                keywords,
                timestamp,
                createdAt: new Date().toISOString()
            }
        });

        allParentDocuments.push(parentDoc);
        console.log(`Created parent transcript for clientId: ${clientId} (${userName}) from ${file}`);
    }

    // Add documents to ParentDocumentRetriever
    // This automatically handles:
    // 1. Parent splitting (1000 chars, 200 overlap)
    // 2. Child splitting (250 chars, 50 overlap)
    // 3. Child embeddings → VectorDB (with clientId metadata)
    // 4. Parent storage → DocStore
    await transcriptRetriever.addDocuments(allParentDocuments);

    // Create BM25 index from parent documents for hybrid search
    createBM25Index(allParentDocuments, 'transcripts');

    console.log(`Successfully ingested ${allParentDocuments.length} parent transcripts with automatic child chunking`);
}