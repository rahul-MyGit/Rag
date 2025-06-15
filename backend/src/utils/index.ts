import { Document } from 'llamaindex';
import { PDFReader } from '@llamaindex/readers/pdf';
import { SimpleDirectoryReader } from '@llamaindex/readers/directory';
import { TextFileReader } from '@llamaindex/readers/text';
import { MarkdownReader } from '@llamaindex/readers/markdown';
import type { AgencyMetadata } from '../types';
import path from 'path';
import fs from 'fs';

/**
 * Extract metadata from transcript filename
 * Format: {name}-{month}-{day}.pdf
 */
export function extractTranscriptMetadata(filePath: string, clientId: 1 | 2): AgencyMetadata {
    const fileName = path.basename(filePath);
    const dateMatch = fileName.match(/(\w+)-(\d{2})-(\d{2})\.pdf/);
    const userId = clientId === 1 ? 'nathan' : 'robert';
    
    return {
        source: fileName,
        type: 'transcript',
        userId,
        clientId,
        fileName,
        date: dateMatch ? `${dateMatch[2]}-${dateMatch[3]}` : undefined,
        filePath
    };
}

/**
 * Extract metadata from document filename
 */
export function extractDocumentMetadata(filePath: string): AgencyMetadata {
    const fileName = path.basename(filePath);
    
    return {
        source: fileName,
        type: 'document',
        fileName,
        filePath
    };
}

/**
 * Load documents from directory with metadata using LlamaIndex readers
 */
export async function loadDocumentsWithMetadata(
    directoryPath: string,
    metadataExtractor: (filePath: string) => AgencyMetadata
): Promise<Document[]> {
    if (!fs.existsSync(directoryPath)) {
        console.warn(`Directory does not exist: ${directoryPath}`);
        return [];
    }

    try {
        // Use SimpleDirectoryReader with proper file type readers
        const reader = new SimpleDirectoryReader();
        const documents = await reader.loadData({
            directoryPath,
            fileExtToReader: {
                pdf: new PDFReader(),
                txt: new TextFileReader(),
                md: new MarkdownReader()
            }
        });

        // Apply custom metadata to each document
        const documentsWithMetadata = documents.map(doc => {
            const filePath = doc.metadata.file_path;
            const customMetadata = metadataExtractor(filePath);
            
            return new Document({
                text: doc.getText(),
                metadata: {
                    ...doc.metadata,
                    ...customMetadata
                }
            });
        });

        console.log(`📚 Loaded ${documentsWithMetadata.length} documents from ${directoryPath}`);
        return documentsWithMetadata;

    } catch (error) {
        console.error(`Failed to load documents from ${directoryPath}:`, error);
        return [];
    }
}

/**
 * Load transcripts from nested directory structure
 */
export async function loadTranscriptsWithClientMetadata(transcriptsPath: string): Promise<Document[]> {
    const allDocs: Document[] = [];
    
    // Load Nathan's transcripts
    const nathanPath = path.join(transcriptsPath, 'nathan');
    if (fs.existsSync(nathanPath)) {
        const nathanDocs = await loadDocumentsWithMetadata(
            nathanPath,
            (filePath) => extractTranscriptMetadata(filePath, 1)
        );
        allDocs.push(...nathanDocs);
        console.log(`📝 Loaded ${nathanDocs.length} Nathan transcripts`);
    }

    // Load Robert's transcripts
    const robertPath = path.join(transcriptsPath, 'robert');
    if (fs.existsSync(robertPath)) {
        const robertDocs = await loadDocumentsWithMetadata(
            robertPath,
            (filePath) => extractTranscriptMetadata(filePath, 2)
        );
        allDocs.push(...robertDocs);
        console.log(`📝 Loaded ${robertDocs.length} Robert transcripts`);
    }

    return allDocs;
}