import { 
    docRetriever, 
    transcriptDenseVectorStore,
    docDenseIndex,
    docSparseIndex,
    transcriptDenseIndex,
    transcriptSparseIndex,
    embeddings
} from "../config/initialize";
import { Document } from "langchain/document";
import { parsePDF, generateSummary, extractKeywords } from "../utils";
import { createDocumentChunks, createTranscriptChunks } from "../utils/chunk";
import { createBM25Index } from "../utils/bm25";
import { documentSparseGenerator, transcriptSparseGenerator } from "../utils/sparse-embeddings";
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
    if (!docRetriever || !docDenseIndex || !docSparseIndex) {
        throw new Error('Document storage systems not initialized');
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

    // Step 1: Extract all text and generate summaries for vocabulary building
    console.log('\n🔧 Phase 1: Building sparse vocabulary and generating summaries...');
    const allTexts: string[] = [];
    const fileDataMap = new Map<string, { text: string, summary: string, keywords: string[] }>();

    for (const filePath of files) {
        try {
            const file = path.basename(filePath);
            console.log(`📄 Processing ${file} for vocabulary and summary...`);
            
            const pdfResult = await parsePDF(filePath, false);
            
            if (typeof pdfResult === 'string' && pdfResult.trim().length > 0) {
                console.log(`📝 Generating summary for ${file}...`);
                const summary = await generateSummary(pdfResult);
                
                console.log(`🔑 Extracting keywords for ${file}...`);
                const keywords = await extractKeywords(pdfResult);
                
                allTexts.push(pdfResult);
                fileDataMap.set(filePath, {
                    text: pdfResult,
                    summary,
                    keywords
                });
                
                console.log(`✅ Processed ${file}: ${pdfResult.length} chars, summary: ${summary.length} chars, ${keywords.length} keywords`);
            }
        } catch (error) {
            console.error(`Error reading ${filePath} for vocabulary:`, error);
        }
    }

    // Build vocabulary for sparse embeddings
    console.log(`🔧 Building document vocabulary from ${allTexts.length} texts...`);
    documentSparseGenerator.buildVocabulary(allTexts);
    console.log(`📊 Document vocabulary size: ${documentSparseGenerator.getVocabularySize()}`);

    // Step 2: Process and store documents
    console.log('\n📚 Phase 2: Processing and storing documents...');
    
    for (const filePath of files) {
        try {
            const file = path.basename(filePath);
            console.log(`\n=== Processing document: ${file} ===`);
            
            const fileData = fileDataMap.get(filePath);
            if (!fileData) {
                console.warn(`No data found for ${file}, skipping`);
                continue;
            }
            
            const { text: fullText, summary, keywords } = fileData;
            
            console.log(`✅ PDF text ready: ${fullText.length} characters from ${file}`);
            
            console.log(`Creating document chunks for ${file}...`);
            const childDocuments = await createDocumentChunks(fullText, file);
            
            // Add summary and keywords to each chunk's metadata
            childDocuments.forEach(doc => {
                doc.metadata.summary = summary;
                doc.metadata.keywords = keywords;
            });
            
            if (childDocuments.length === 0) {
                console.warn(`No chunks created for ${file}, skipping`);
                continue;
            }

            console.log(`\n=== HYBRID STORAGE FOR ${file} ===`);
            console.log(`Total chunks: ${childDocuments.length}`);
            console.log(`Strategy: Dense + Sparse embeddings in separate Pinecone indexes`);

            // Store in dense vector store (LangChain retriever)
            console.log(`📊 Storing ${childDocuments.length} chunks in dense index...`);
            await docRetriever.addDocuments(childDocuments);

            // Store in sparse vector store
            console.log(`📈 Storing ${childDocuments.length} chunks in sparse index...`);
            await storeSparseEmbeddings(childDocuments, 'documents');

            createBM25Index(childDocuments, 'documents');

            console.log(`✅ Successfully processed ${file} with hybrid storage`);
            console.log(`=== END HYBRID STORAGE ===\n`);
            
        } catch (error) {
            console.error(`❌ Error processing file ${filePath}:`, error);
            continue;
        }
    }

    console.log('\n🎉 Document ingestion with hybrid search completed successfully!');
}

// Helper function to store sparse embeddings in Pinecone
async function storeSparseEmbeddings(documents: Document[], type: 'documents' | 'transcripts'): Promise<void> {
    const sparseIndex = type === 'documents' ? docSparseIndex : transcriptSparseIndex;
    const sparseGenerator = type === 'documents' ? documentSparseGenerator : transcriptSparseGenerator;
    
    const vectors = [];
    
    for (const doc of documents) {
        const sparseVector = sparseGenerator.generateSparseEmbedding(doc.pageContent);
        
        if (sparseVector.indices.length === 0) {
            console.warn(`⚠️ Empty sparse vector generated, skipping chunk`);
            continue; // Skip empty vectors
        }
        
        // Create minimal dense vector for sparse index compatibility
        const denseVector = new Array(10000).fill(0);
        denseVector[0] = 0.001; // Minimal non-zero value
        
        vectors.push({
            id: doc.metadata.id || `${type}_${Date.now()}_${Math.random()}`,
            values: denseVector,
            sparseValues: {
                indices: sparseVector.indices,
                values: sparseVector.values
            },
            metadata: {
                text: doc.pageContent.substring(0, 300), // Much smaller text preview
                summary: doc.metadata.summary?.substring(0, 200) || '', // Shorter summary
                keywords: doc.metadata.keywords?.slice(0, 5).join(',') || '', // Fewer keywords, no spaces
                type: doc.metadata.type,
                fileName: doc.metadata.fileName,
                chunkIndex: doc.metadata.chunkIndex,
                clientId: doc.metadata.clientId || 0
            }
        });
    }
    
    if (vectors.length > 0) {
        // Process in batches to avoid metadata size limits
        const batchSize = 50; // Smaller batches
        for (let i = 0; i < vectors.length; i += batchSize) {
            const batch = vectors.slice(i, i + batchSize);
            await sparseIndex.upsert(batch);
            console.log(`✅ Stored batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(vectors.length/batchSize)}: ${batch.length} sparse embeddings`);
            
            // Small delay between batches
            if (i + batchSize < vectors.length) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        console.log(`✅ Completed storing ${vectors.length} sparse embeddings in ${type} index`);
    }
}

export async function ingestTranscripts(): Promise<void> {
    if (!transcriptDenseVectorStore || !transcriptDenseIndex || !transcriptSparseIndex) {
        throw new Error('Transcript storage systems not initialized');
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

    // Step 1: Extract all text and generate summaries for vocabulary building
    console.log('\n🔧 Phase 1: Building transcript sparse vocabulary and generating summaries...');
    const allTexts: string[] = [];
    const transcriptDataMap = new Map<string, { text: string, userName: string, clientId: number, summary: string, keywords: string[] }>();

    for (const filePath of files) {
        try {
            const file = path.basename(filePath);
            console.log(`📄 Processing transcript ${file} for vocabulary and summary...`);
            
            const pdfResult = await parsePDF(filePath, false);
            
            if (typeof pdfResult === 'string' && pdfResult.trim().length > 0) {
                const clientId = file.toLowerCase().includes('nathan') ? 1 : 
                               file.toLowerCase().includes('robert') ? 2 : 0;
                const userName = clientId === 1 ? 'nathan' : 
                               clientId === 2 ? 'robert' : 'unknown';

                console.log(`📝 Generating transcript summary for ${userName}...`);
                const summary = await generateSummary(pdfResult);
                
                console.log(`🔑 Extracting transcript keywords for ${userName}...`);
                const keywords = await extractKeywords(pdfResult);

                allTexts.push(pdfResult);
                transcriptDataMap.set(filePath, {
                    text: pdfResult,
                    userName,
                    clientId,
                    summary,
                    keywords
                });
                
                console.log(`✅ Processed ${file}: ${pdfResult.length} chars, summary: ${summary.length} chars, ${keywords.length} keywords`);
            }
        } catch (error) {
            console.error(`Error reading ${filePath} for vocabulary:`, error);
        }
    }

    // Build vocabulary for sparse embeddings
    console.log(`🔧 Building transcript vocabulary from ${allTexts.length} texts...`);
    transcriptSparseGenerator.buildVocabulary(allTexts);
    console.log(`📊 Transcript vocabulary size: ${transcriptSparseGenerator.getVocabularySize()}`);

    // Step 2: Process and store transcripts
    console.log('\n💬 Phase 2: Processing and storing transcripts...');

    for (const filePath of files) {
        try {
            const file = path.basename(filePath);
            console.log(`\n=== Processing transcript: ${file} ===`);
            
            const transcriptData = transcriptDataMap.get(filePath);
            if (!transcriptData) {
                console.warn(`No data found for ${file}, skipping`);
                continue;
            }

            const { text, userName, clientId, summary, keywords } = transcriptData;
            console.log(`✅ Processing ${text.length} characters for user ${userName} (clientId: ${clientId})`);

            console.log(`Creating transcript chunks for ${userName}...`);
            const childDocuments = await createTranscriptChunks(text, file, userName);
            
            childDocuments.forEach(doc => {
                doc.metadata.clientId = clientId;
                doc.metadata.userId = userName;
                doc.metadata.summary = summary;
                doc.metadata.keywords = keywords;
            });

            if (childDocuments.length === 0) {
                console.warn(`No chunks created for ${file}, skipping`);
                continue;
            }

            console.log(`\n=== HYBRID TRANSCRIPT STORAGE FOR ${file} ===`);
            console.log(`User: ${userName} (clientId: ${clientId})`);
            console.log(`Total chunks: ${childDocuments.length}`);
            console.log(`Strategy: Dense + Sparse embeddings in separate Pinecone indexes`);

            // Store in dense vector store
            console.log(`📊 Storing ${childDocuments.length} chunks in dense transcript index...`);
            await transcriptDenseVectorStore.addDocuments(childDocuments);

            // Store in sparse vector store  
            console.log(`📈 Storing ${childDocuments.length} chunks in sparse transcript index...`);
            await storeSparseEmbeddings(childDocuments, 'transcripts');

            createBM25Index(childDocuments, 'transcripts');

            console.log(`✅ Successfully processed ${file} for ${userName} with hybrid storage`);
            console.log(`=== END HYBRID TRANSCRIPT STORAGE ===\n`);
            
        } catch (error) {
            console.error(`❌ Error processing file ${filePath}:`, error);
            continue;
        }
    }

    console.log('\n🎉 Transcript ingestion with hybrid search completed successfully!');
}
