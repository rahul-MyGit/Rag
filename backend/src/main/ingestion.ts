import { docRetriever, transcriptVectorStore } from "../config/initialize";
import { Document } from "langchain/document";
import { parsePDF, generateSummary, extractKeywords } from "../utils";
import { createDocumentChunks, createTranscriptChunks } from "../utils/chunk";
import { createBM25Index } from "../utils/bm25";
import fs from 'fs';
import path from 'path';

function getAllPdfFiles(dir: string): string[] {
    console.log(`Scanning directory: ${dir}`);
    let results: string[] = [];
    
    if (!fs.existsSync(dir)) {
        console.error(`Directory does not exist: ${dir}`);
        return results;
    }
    
    try {
        const list = fs.readdirSync(dir);
        console.log(`Found ${list.length} items in directory`);
        
        list.forEach(function(file) {
            const filePath = path.resolve(dir, file);
            console.log(`Checking: ${filePath}`);
            
            try {
                const stat = fs.statSync(filePath);
                if (stat && stat.isDirectory()) {
                    console.log(`Recursing into subdirectory: ${filePath}`);
                    results = results.concat(getAllPdfFiles(filePath));
                } else if (file.toLowerCase().endsWith('.pdf')) {
                    console.log(`Found PDF: ${filePath}`);
                    results.push(filePath);
                }
            } catch (statError) {
                console.error(`Error accessing ${filePath}:`, statError);
            }
        });
    } catch (readError) {
        console.error(`Error reading directory ${dir}:`, readError);
    }
    
    return results;
}

export async function ingestDocuments(): Promise<void> {
    if (!docRetriever) {
        throw new Error('Document retriever not initialized');
    }

    const docsPath = path.resolve('D:\\100xSuper30\\rag-playground\\docs_for_test');
    console.log(`Looking for documents in: ${docsPath}`);
    
    if (!fs.existsSync(docsPath)) {
        console.error('Documents directory not found:', docsPath);
        return;
    }

    const files = getAllPdfFiles(docsPath);
    console.log(`Found ${files.length} document files to process`);

    if (files.length === 0) {
        console.log('No PDF files found to process');
        return;
    }

    for (const filePath of files) {
        try {
            const file = path.basename(filePath);
            console.log(`\n=== Processing document: ${file} ===`);
            
            console.log(`Parsing entire PDF: ${file}...`);
            const pdfResult = await parsePDF(filePath, false);
            
            if (typeof pdfResult !== 'string') {
                console.warn(`Expected string from parsePDF but got array for ${file}, skipping`);
                continue;
            }
            
            const fullText = pdfResult;
            
            if (!fullText || fullText.trim().length === 0) {
                console.warn(`No text extracted from ${file}, skipping`);
                continue;
            }
            
            console.log(`✅ PDF parsed successfully: ${fullText.length} characters from all pages of ${file}`);
            
            console.log(`Creating optimized document chunks for ${file}...`);
            const childDocuments = await createDocumentChunks(fullText, file);
            
            console.log(`\n=== OPTIMIZED DOCUMENT CHUNK SUMMARY FOR ${file} ===`);
            console.log(`Original PDF size: ${fullText.length} characters`);
            console.log(`Parent chunks: ${childDocuments.length / 2} (stored in docstore)`);
            console.log(`Child chunks: ${childDocuments.length} (exactly 2 per parent)`);
            console.log(`Strategy: Each parent → 2 children → embed raw content (no summaries)`);
            console.log(`Storage: Raw child content in Pinecone, parent content in docstore`);
            console.log(`Benefits: No LLM costs, no information loss, faster processing`);
            console.log(`=== END OPTIMIZED DOCUMENT CHUNK SUMMARY ===\n`);

            if (childDocuments.length === 0) {
                console.warn(`No chunks created for ${file}, skipping`);
                continue;
            }

            console.log(`Adding ${childDocuments.length} child chunks to retriever for ${file}...`);
            await docRetriever.addDocuments(childDocuments);

            createBM25Index(childDocuments, 'documents');

            console.log(`✅ Successfully processed ${file} with ${childDocuments.length} chunks`);
            
        } catch (error) {
            console.error(`❌ Error processing file ${filePath}:`, error);
            continue;
        }
    }

    console.log('\n🎉 Document ingestion completed successfully!');
}

export async function ingestTranscripts(): Promise<void> {
    if (!transcriptVectorStore) {
        throw new Error('Transcript vector store not initialized');
    }

    const transPath = path.resolve('D:\\100xSuper30\\rag-playground\\transcriptions_for_test');
    console.log(`Looking for transcripts in: ${transPath}`);
    
    if (!fs.existsSync(transPath)) {
        console.error('Transcripts directory not found:', transPath);
        return;
    }

    const files = getAllPdfFiles(transPath);
    console.log(`Found ${files.length} transcript files to process`);

    if (files.length === 0) {
        console.log('No PDF files found to process');
        return;
    }

    for (const filePath of files) {
        try {
            const file = path.basename(filePath);
            console.log(`\n=== Processing transcript: ${file} ===`);
            
            const pdfResult = await parsePDF(filePath, false);
            
            if (typeof pdfResult !== 'string') {
                console.warn(`Expected string from parsePDF but got array for ${file}, skipping`);
                continue;
            }
            
            const text = pdfResult;
            
            if (!text || text.trim().length === 0) {
                console.warn(`No text extracted from ${file}, skipping`);
                continue;
            }
            
            const clientId = file.toLowerCase().includes('nathan') ? 1 : 
                            file.toLowerCase().includes('robert') ? 2 : 0;
            
            const userName = clientId === 1 ? 'nathan' : 
                            clientId === 2 ? 'robert' : 'unknown';

            console.log(`Extracted ${text.length} characters for user ${userName} (clientId: ${clientId})`);

            console.log(`Creating transcript chunks for ${userName}...`);
            const childDocuments = await createTranscriptChunks(text, file, userName);
            
            childDocuments.forEach(doc => {
                doc.metadata.clientId = clientId;
                doc.metadata.userId = userName;
            });

            console.log(`\n=== TRANSCRIPT CHUNK LOGGING FOR ${file} ===`);
            console.log(`User: ${userName} (clientId: ${clientId})`);
            console.log(`Total chunks created: ${childDocuments.length}`);
            if (text.length < 1500) {
                console.log(`Short transcript - embedded as single chunk`);
            } else {
                console.log(`Longer transcript - simple chunking without parent-child hierarchy`);
            }
            console.log(`Content embedded directly (no summaries for transcripts)`);
            console.log(`=== END TRANSCRIPT CHUNK LOGGING ===\n`);

            if (childDocuments.length === 0) {
                console.warn(`No chunks created for ${file}, skipping`);
                continue;
            }

            console.log(`Adding ${childDocuments.length} chunks to vector store for ${userName}...`);
            await transcriptVectorStore.addDocuments(childDocuments);

            createBM25Index(childDocuments, 'transcripts');

            console.log(`✅ Successfully processed ${file} for ${userName} with ${childDocuments.length} chunks`);
            
        } catch (error) {
            console.error(`❌ Error processing file ${filePath}:`, error);
            continue;
        }
    }

    console.log('\n🎉 Transcript ingestion completed successfully!');
}
