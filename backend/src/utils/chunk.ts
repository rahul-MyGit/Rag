import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { v4 as uuidv4 } from 'uuid';
import { CONFIG } from "../config";
import { generateSummary, extractKeywords } from "./index";
import { type ChunkMetadata } from "../types";

export async function createDocumentChunks(text: string, fileName: string): Promise<Document[]> {
    const config = CONFIG.CHUNKING.DOCUMENT;

    // Parent splitter - larger chunks for documents
    const parentSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: config.PARENT_SIZE,
        chunkOverlap: config.PARENT_OVERLAP,
        separators: ['\n\n', '\n', '. ', '; ', ', ', ' ', '']
    });

    // Child splitter - medium chunks for embedding
    const childSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: config.CHILD_SIZE,
        chunkOverlap: config.CHILD_OVERLAP,
        separators: ['\n\n', '\n', '. ', ' ', '']
    });

    const parentChunks = await parentSplitter.createDocuments([text]);
    const documents: Document[] = [];

    for (let i = 0; i < parentChunks.length; i++) {
        const parentId = uuidv4();
        const parentChunk = parentChunks[i];

        // Generate summary and keywords for parent chunk
        const summary = await generateSummary(parentChunk?.pageContent || '');
        const keywords = extractKeywords(parentChunk?.pageContent || '');

        // Create child chunks from parent chunk
        const childChunks = await childSplitter.createDocuments([parentChunk?.pageContent || '']);

        for (let j = 0; j < childChunks.length; j++) {
            const childChunk = childChunks[j];
            const metadata: ChunkMetadata = {
                source: `${fileName}_parent_${i}_child_${j}`,
                type: 'document',
                parentId,
                chunkIndex: j,
                fileName,
                summary,
                keywords
            };

            documents.push(new Document({
                pageContent: childChunk?.pageContent || '',
                metadata
            }));
        }
    }

    return documents;
}

export async function createTranscriptChunks(text: string, fileName: string, userId: string): Promise<Document[]> {
    const config = CONFIG.CHUNKING.TRANSCRIPT;

    // For transcripts, use smaller chunks to preserve conversational context
    const parentSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: config.PARENT_SIZE,
        chunkOverlap: config.PARENT_OVERLAP,
        separators: ['\n\n', '\n', '. ', '? ', '! ', ', ', ' ', '']
    });

    const childSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: config.CHILD_SIZE,
        chunkOverlap: config.CHILD_OVERLAP,
        separators: ['\n', '. ', '? ', '! ', ' ', '']
    });

    const parentChunks = await parentSplitter.createDocuments([text]);
    const documents: Document[] = [];

    for (let i = 0; i < parentChunks.length; i++) {
        const parentId = uuidv4();
        const parentChunk = parentChunks[i];

        // Generate summary and keywords
        const summary = await generateSummary(parentChunk?.pageContent || '');
        const keywords = extractKeywords(parentChunk?.pageContent || '');

        // Extract timestamp if present (format: name-month-day)
        const timestampMatch = fileName.match(/(\d{2}-\d{2})/);
        const timestamp = timestampMatch ? timestampMatch[1] : undefined;

        const childChunks = await childSplitter.createDocuments([parentChunk?.pageContent || '']);

        for (let j = 0; j < childChunks.length; j++) {
            const childChunk = childChunks[j];
            const metadata: ChunkMetadata = {
                source: `${fileName}_parent_${i}_child_${j}`,
                type: 'transcript',
                userId,
                parentId,
                chunkIndex: j,
                fileName,
                summary,
                keywords,
                timestamp
            };

            documents.push(new Document({
                pageContent: childChunk?.pageContent || '',
                metadata
            }));
        }
    }

    return documents;
}