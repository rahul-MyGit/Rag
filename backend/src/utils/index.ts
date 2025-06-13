import { llm } from "../config/initialize";
import fs from 'fs';
import pdfParse from 'pdf-parse';
import * as natural from 'natural';
import { removeStopwords } from 'stopword';
import { stemmer } from 'stemmer';

export async function parsePDF(filePath: string): Promise<string> {
    try {
        const pdfBuffer = fs.readFileSync(filePath);
        const data = await pdfParse(pdfBuffer);
        return data.text;
    } catch (error) {
        console.error(`Error parsing PDF ${filePath}:`, error);
        throw error;
    }
}

export async function generateSummary(text: string): Promise<string> {
    try {
        const prompt = `Provide a concise summary of the following text in 2-3 sentences, focusing on key topics and main points:
  
  ${text.substring(0, 2000)}...
  
  Summary:`;

        const response = await llm.invoke(prompt);
        return response.content as string;
    } catch (error) {
        console.error('Error generating summary:', error);
        return 'Summary generation failed';
    }
}

export function extractKeywords(text: string): string[] {
    try {
        // Tokenize and clean
        const tokenizer = new natural.WordTokenizer();
        const tokens = tokenizer.tokenize(text.toLowerCase()) || [];

        // Remove stopwords
        const filteredTokens = removeStopwords(tokens);

        // Stem and get unique keywords
        const stemmedTokens = filteredTokens.map(token => stemmer(token));

        // Simple frequency counting and get top keywords
        const frequency: { [key: string]: number } = {};
        stemmedTokens.forEach(token => {
            frequency[token] = (frequency[token] || 0) + 1;
        });

        const topWords = Object.entries(frequency)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10)
            .map(([token]) => token);

        return topWords;
    } catch (error) {
        console.error('Error extracting keywords:', error);
        return [];
    }
}