import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { v4 as uuidv4 } from 'uuid';
import { CONFIG } from "../config";
import { generateSummary, extractKeywords } from "./index";
import { type ChunkMetadata } from "../types";

export async function createDocumentChunks(text: string, fileName: string): Promise<Document[]> {
    const config = CONFIG.CHUNKING.DOCUMENT;

    console.log(`Starting document chunking for ${fileName} (${text.length} characters)`);

    const parentSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: config.PARENT_SIZE,
        chunkOverlap: config.PARENT_OVERLAP,
        separators: ['\n\n', '\n', '. ', '; ', ', ', ' ', '']
    });

    console.log(`Creating parent chunks for ${fileName}...`);
    const parentChunks = await parentSplitter.createDocuments([text]);
    console.log(`Created ${parentChunks.length} parent chunks for ${fileName}`);

    const allDocuments: Document[] = [];

    for (let parentIndex = 0; parentIndex < parentChunks.length; parentIndex++) {
        const parentChunk = parentChunks[parentIndex];
        const parentContent = parentChunk?.pageContent || '';
        const parentId = uuidv4();

        console.log(`Processing parent chunk ${parentIndex + 1}/${parentChunks.length} for ${fileName} (${parentContent.length} chars)`);

        const midPoint = Math.floor(parentContent.length / 2);
        const overlapSize = config.CHILD_OVERLAP;
        
        const childChunks = [
            {
                pageContent: parentContent.slice(0, midPoint + overlapSize),
                metadata: {}
            },
            {
                pageContent: parentContent.slice(midPoint - overlapSize),
                metadata: {}
            }
        ];

        console.log(`  Created exactly 2 child chunks from parent ${parentIndex + 1}`);
        console.log(`    Child 1: ${childChunks[0]?.pageContent.length} chars`);
        console.log(`    Child 2: ${childChunks[1]?.pageContent.length} chars`);

        for (let childIndex = 0; childIndex < 2; childIndex++) {
            const childChunk = childChunks[childIndex];
            const childContent = childChunk?.pageContent || '';
            
            console.log(`    Processing child ${childIndex + 1}/2 of parent ${parentIndex + 1} (${childContent.length} chars)`);
            
            const childKeywords = extractKeywords(childContent);

            const metadata: ChunkMetadata = {
                source: `${fileName}_parent_${parentIndex}_child_${childIndex}`,
                type: 'document',
                parentId,
                chunkIndex: childIndex,
                fileName,
                summary: undefined,
                keywords: await childKeywords,
                originalContent: childContent
            };

            allDocuments.push(new Document({
                pageContent: childContent,
                metadata
            }));
        }

        console.log(`  Completed parent chunk ${parentIndex + 1} → 2 child chunks, adding delay...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    console.log(`✅ Document chunking completed for ${fileName}:`);
    console.log(`   📊 ${parentChunks.length} parent chunks → ${allDocuments.length} child chunks (2 per parent)`);
    console.log(`   💾 Parent chunks stored in docstore, raw child content embedded in Pinecone`);
    console.log(`   🚀 Benefits: Zero LLM costs, no information loss, faster processing`);
    
    return allDocuments;
}

export async function createTranscriptChunks(text: string, fileName: string, userId: string): Promise<Document[]> {
    const config = CONFIG.CHUNKING.TRANSCRIPT;
    
    console.log(`Processing transcript for ${userId}: ${text.length} characters`);

    if (text.length < 1500) {
        console.log(`Short transcript detected, embedding entire content`);
        
        const timestampMatch = fileName.match(/(\d{2}-\d{2})/);
        const timestamp = timestampMatch ? timestampMatch[1] : undefined;

        const metadata: ChunkMetadata = {
            source: `${fileName}_full`,
            type: 'transcript',
            userId,
            parentId: undefined,
            chunkIndex: 0,
            fileName,
            keywords: await extractKeywords(text),
            timestamp,
            originalContent: text
        };

        return [new Document({
            pageContent: text,
            metadata
        })];
    }

    console.log(`Longer transcript detected, using simple chunking`);
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: config.CHILD_SIZE * 2,
        chunkOverlap: config.CHILD_OVERLAP * 2,
        separators: ['\n\n', '\n', '. ', '? ', '! ', ', ', ' ', '']
    });

    const chunks = await splitter.createDocuments([text]);
    const documents: Document[] = [];

    const timestampMatch = fileName.match(/(\d{2}-\d{2})/);
    const timestamp = timestampMatch ? timestampMatch[1] : undefined;

    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const chunkContent = chunk?.pageContent || '';
        
        console.log(`  Processing transcript chunk ${i + 1}/${chunks.length}`);

        const metadata: ChunkMetadata = {
            source: `${fileName}_chunk_${i}`,
            type: 'transcript',
            userId,
            parentId: undefined,
            chunkIndex: i,
            fileName,
            keywords: await extractKeywords(chunkContent),
            timestamp,
            originalContent: chunkContent
        };

        documents.push(new Document({
            pageContent: chunkContent,
            metadata
        }));
    }

    console.log(`Created ${documents.length} simple transcript chunks for ${fileName}`);
    return documents;
}