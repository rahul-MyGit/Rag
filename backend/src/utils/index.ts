import { llm } from "../config/initialize";
import fs from 'fs';
// import * as natural from 'natural';
import { removeStopwords } from 'stopword';
import { stemmer } from 'stemmer';
import path from 'path';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

type PageData = {
    content: string;
    pageNumber: number;
    metadata: Record<string, any>;
};

export async function parsePDF(filePath: string, splitPages: true): Promise<PageData[]>;
export async function parsePDF(filePath: string, splitPages: false): Promise<string>;
export async function parsePDF(filePath: string, splitPages?: boolean): Promise<string | PageData[]>;
export async function parsePDF(filePath: string, splitPages: boolean = false): Promise<string | PageData[]> {
    try {
        const absolutePath = path.resolve(filePath);
        
        if (!fs.existsSync(absolutePath)) {
            throw new Error(`PDF file not found: ${absolutePath}`);
        }

        console.log(`Reading PDF from: ${absolutePath}`);
        
        const loader = new PDFLoader(absolutePath, {
            splitPages: splitPages,
            parsedItemSeparator: " " 
        });
        
        const docs = await loader.load();
        console.log(`Extracted ${docs.length} ${splitPages ? 'pages' : 'document(s)'} from PDF`);
        
        if (!docs || docs.length === 0) {
            throw new Error('No content extracted from PDF');
        }

        if (splitPages) {
            const pages: PageData[] = docs.map((doc, index) => ({
                content: doc.pageContent,
                pageNumber: index + 1,
                metadata: doc.metadata
            }));
            console.log(`Successfully extracted ${pages.length} pages from PDF`);
            return pages;
        } else {
            const text = docs.map(doc => doc.pageContent).join('\n\n');
            
            if (!text || text.trim().length === 0) {
                console.warn(`No text content found in PDF: ${absolutePath}`);
                return '';
            }
            
            console.log(`Successfully extracted ${text.length} characters from PDF`);
            return text;
        }
        
    } catch (error) {
        console.error(`Error parsing PDF ${filePath}:`, error);
        throw new Error(`Failed to parse PDF: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

export async function generateSummary(text: string): Promise<string> {
    try {
        const maxChunkSize = 4000;
        const chunks = [];
        
        for (let i = 0; i < text.length; i += maxChunkSize) {
            chunks.push(text.slice(i, i + maxChunkSize));
        }

        const summaries = [];
        for (const chunk of chunks) {
            const prompt = `Provide a concise summary of the following text in 2-3 sentences, focusing on key topics and main points:
  
  ${chunk}
  
  Summary:`;

            try {
                const response = await llm.invoke(prompt);
                summaries.push(response.content as string);
                await new Promise(resolve => setTimeout(resolve, 1000));
            } catch (error) {
                console.warn('Error processing chunk:', error);
                summaries.push(`Section containing ${chunk.length} characters with key information.`);
            }
        }

        if (summaries.length === 1) {
            return summaries[0] || 'No summary available';
        }

        const finalPrompt = `Combine these summaries into one concise summary (2-3 sentences):
  
  ${summaries.join('\n\n')}
  
  Final Summary:`;

        try {
            const response = await llm.invoke(finalPrompt);
            return response.content as string;
        } catch (error) {
            console.warn('Error creating final summary:', error);
            return summaries.join(' ');
        }
    } catch (error) {
        console.error('Error generating summary:', error);
        return `Document containing ${text.length} characters with key information about the topic.`;
    }
}

export async function extractKeywords(text: string): Promise<string[]> {
    try {
        const maxChunkSize = 4000;
        const chunks = [];
        
        for (let i = 0; i < text.length; i += maxChunkSize) {
            chunks.push(text.slice(i, i + maxChunkSize));
        }

        const allKeywords = new Set<string>();
        
        for (const chunk of chunks) {
            try {
                const splitter = new RecursiveCharacterTextSplitter({
                    chunkSize: 1000,
                    chunkOverlap: 200,
                });
                
                const subChunks = await splitter.splitText(chunk);
                
                const words = subChunks
                    .join(' ')
                    .toLowerCase()
                    .split(/\s+/)
                    .filter((word: string) => word.length > 2);
                    
                const filteredWords = removeStopwords(words);
                
                const stemmedWords = filteredWords.map(word => stemmer(word));
                
                stemmedWords.forEach(word => allKeywords.add(word));
                
                await new Promise(resolve => setTimeout(resolve, 500));
            } catch (error) {
                console.warn('Error processing chunk for keywords:', error);
            }
        }
        
        const keywords = Array.from(allKeywords);
        return keywords.slice(0, 10);
    } catch (error) {
        console.error('Error extracting keywords:', error);
        return ['document', 'information', 'content'];
    }
}