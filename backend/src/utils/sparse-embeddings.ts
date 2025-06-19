import { stemmer } from 'stemmer';
import { removeStopwords } from 'stopword';

interface SparseVector {
    indices: number[];
    values: number[];
}

class SparseEmbeddingGenerator {
    private vocabulary: Map<string, number> = new Map();
    private documentFrequencies: Map<string, number> = new Map();
    private totalDocuments: number = 0;
    private vocabularySize: number = 0;

    private preprocessText(text: string): string[] {
        // Convert to lowercase and remove special characters
        const cleanText = text.toLowerCase().replace(/[^\w\s]/g, ' ');
        
        // Tokenize
        const tokens = cleanText.split(/\s+/).filter(token => token.length > 2);
        
        // Remove stopwords
        const withoutStopwords = removeStopwords(tokens);
        
        // Apply stemming
        const stemmed = withoutStopwords.map(token => stemmer(token));
        
        return stemmed;
    }

    buildVocabulary(documents: string[]): void {
        console.log(`🔧 Building vocabulary from ${documents.length} documents...`);
        
        const termCounts = new Map<string, number>();
        
        // Count term frequencies across all documents
        for (const doc of documents) {
            const tokens = this.preprocessText(doc);
            const uniqueTokens = new Set(tokens);
            
            for (const token of uniqueTokens) {
                termCounts.set(token, (termCounts.get(token) || 0) + 1);
            }
        }
        
        // Filter terms that appear in at least 2 documents but not more than 80% of documents
        const minFreq = Math.max(2, Math.floor(documents.length * 0.01));
        const maxFreq = Math.floor(documents.length * 0.8);
        
        console.log(`📊 Vocabulary filtering: minFreq=${minFreq}, maxFreq=${maxFreq}`);
        console.log(`📊 Total unique terms found: ${termCounts.size}`);
        
        let vocabIndex = 0;
        let acceptedTerms = 0;
        let rejectedTooRare = 0;
        let rejectedTooCommon = 0;
        
        for (const [term, freq] of termCounts.entries()) {
            if (freq >= minFreq && freq <= maxFreq) {
                this.vocabulary.set(term, vocabIndex++);
                this.documentFrequencies.set(term, freq);
                acceptedTerms++;
            } else if (freq < minFreq) {
                rejectedTooRare++;
            } else {
                rejectedTooCommon++;
            }
        }
        
        console.log(`📊 Vocabulary stats: accepted=${acceptedTerms}, tooRare=${rejectedTooRare}, tooCommon=${rejectedTooCommon}`);
        console.log(`📊 Sample accepted terms: ${Array.from(this.vocabulary.keys()).slice(0, 10).join(', ')}`);
        console.log(`📊 Sample term frequencies: ${Array.from(this.documentFrequencies.entries()).slice(0, 5).map(([t, f]) => `${t}:${f}`).join(', ')}`);
        
        this.totalDocuments = documents.length;
        this.vocabularySize = this.vocabulary.size;
        
        console.log(`✅ Vocabulary built: ${this.vocabularySize} terms from ${this.totalDocuments} documents`);
        console.log(`📊 Term frequency range: ${minFreq} to ${maxFreq}`);
    }

    generateSparseEmbedding(text: string): SparseVector {
        const tokens = this.preprocessText(text);
        
        // Calculate term frequencies in this document
        const termFreqs = new Map<string, number>();
        for (const token of tokens) {
            termFreqs.set(token, (termFreqs.get(token) || 0) + 1);
        }
        
        const indices: number[] = [];
        const values: number[] = [];
        
        // Calculate TF-IDF for each term
        for (const [term, tf] of termFreqs.entries()) {
            if (this.vocabulary.has(term)) {
                const index = this.vocabulary.get(term)!;
                const df = this.documentFrequencies.get(term) || 1;
                
                // TF-IDF calculation
                const tfScore = Math.log(1 + tf); // Log-normalized TF
                const idfScore = Math.log(this.totalDocuments / df); // IDF
                const tfidfScore = tfScore * idfScore;
                
                if (tfidfScore > 0.01) { // Filter very low scores
                    indices.push(index);
                    values.push(tfidfScore);
                }
            }
        }
        
        // Normalize the vector
        const magnitude = Math.sqrt(values.reduce((sum, val) => sum + val * val, 0));
        if (magnitude > 0) {
            for (let i = 0; i < values.length; i++) {
                values[i] = (values[i] ?? 0) / magnitude;
            }
        }
        
        return { indices, values };
    }

    getVocabularySize(): number {
        return this.vocabularySize;
    }

    exportVocabulary(): { vocabulary: Record<string, number>, documentFrequencies: Record<string, number> } {
        return {
            vocabulary: Object.fromEntries(this.vocabulary),
            documentFrequencies: Object.fromEntries(this.documentFrequencies)
        };
    }

    importVocabulary(data: { vocabulary: Record<string, number>, documentFrequencies: Record<string, number>, totalDocuments: number }): void {
        this.vocabulary = new Map(Object.entries(data.vocabulary));
        this.documentFrequencies = new Map(Object.entries(data.documentFrequencies));
        this.totalDocuments = data.totalDocuments;
        this.vocabularySize = this.vocabulary.size;
        
        console.log(`📥 Imported vocabulary: ${this.vocabularySize} terms, ${this.totalDocuments} documents`);
    }
}

// Global instances for documents and transcripts
export const documentSparseGenerator = new SparseEmbeddingGenerator();
export const transcriptSparseGenerator = new SparseEmbeddingGenerator();

export type { SparseVector }; 