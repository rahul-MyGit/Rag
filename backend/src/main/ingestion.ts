import { VectorStoreIndex, Document } from 'llamaindex';
import { CONFIG } from '../config';
import { pinecone } from '../config/initialize';
import { 
    loadDocumentsWithMetadata, 
    loadTranscriptsWithClientMetadata,
    extractDocumentMetadata 
} from '../utils';
import path from 'path';
import fs from 'fs';

export async function buildDocumentIndex(): Promise<VectorStoreIndex | null> {
    console.log('📚 Building document index...');

    const docsPath = path.resolve(CONFIG.PATHS.DOCS_DIR);
    if (!fs.existsSync(docsPath)) {
        console.warn(`Documents directory not found: ${docsPath}`);
        return null;
    }

    const documents = await loadDocumentsWithMetadata(
        docsPath,
        extractDocumentMetadata
    );
    
    if (documents.length === 0) {
        console.warn('No documents found to index');
        return null;
    }

    const documentIndex = await VectorStoreIndex.fromDocuments(documents);

    console.log(`✅ Built document index with ${documents.length} documents`);
    return documentIndex;
}

export async function buildTranscriptIndex(): Promise<VectorStoreIndex | null> {
    console.log('📝 Building transcript index...');

    const transcriptsPath = path.resolve(CONFIG.PATHS.TRANSCRIPTS_DIR);
    if (!fs.existsSync(transcriptsPath)) {
        console.warn(`Transcripts directory not found: ${transcriptsPath}`);
        return null;
    }

    const transcriptDocs = await loadTranscriptsWithClientMetadata(transcriptsPath);
    
    if (transcriptDocs.length === 0) {
        console.warn('No transcripts found to index');
        return null;
    }

    const transcriptIndex = await VectorStoreIndex.fromDocuments(transcriptDocs);

    console.log(`✅ Built transcript index with ${transcriptDocs.length} transcripts`);
    return transcriptIndex;
}
