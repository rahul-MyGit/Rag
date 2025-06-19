import { Document } from "langchain/document";
import { CONFIG } from "../config";
import { 
    docDenseIndex, 
    docSparseIndex, 
    transcriptDenseIndex, 
    transcriptSparseIndex,
    embeddings 
} from "../config/initialize";
import { documentSparseGenerator, transcriptSparseGenerator, type SparseVector } from "./sparse-embeddings";

interface HybridSearchResult {
    document: Document;
    denseScore: number;
    sparseScore: number;
    combinedScore: number;
}

interface PineconeMatch {
    id: string;
    score: number;
    metadata?: Record<string, any>;
}

export async function performPineconeHybridSearch(
    query: string,
    type: 'documents' | 'transcripts',
    userId?: number,
    topK: number = 10
): Promise<HybridSearchResult[]> {
    console.log(`\n=== PINECONE HYBRID SEARCH ===`);
    console.log(`Query: "${query}"`);
    console.log(`Type: ${type}, UserId: ${userId}, TopK: ${topK}`);

    try {
        // Generate dense embedding
        const denseVector = await embeddings.embedQuery(query);
        console.log(`✅ Dense embedding generated: ${denseVector.length} dimensions`);

        // Generate sparse embedding
        const sparseGenerator = type === 'documents' ? documentSparseGenerator : transcriptSparseGenerator;
        const sparseVector = sparseGenerator.generateSparseEmbedding(query);
        console.log(`✅ Sparse embedding generated: ${sparseVector.indices.length} non-zero values`);

        // Prepare search parameters
        const searchParams: any = {
            topK: topK * 2, // Get more results to ensure good coverage
            includeMetadata: true,
            includeValues: false
        };

        // Add user filter for transcripts
        if (type === 'transcripts' && userId) {
            searchParams.filter = { clientId: userId };
        }

        // Perform dense search
        const denseIndex = type === 'documents' ? docDenseIndex : transcriptDenseIndex;
        const denseResults = await denseIndex.query({
            ...searchParams,
            vector: denseVector
        });

        console.log(`📊 Dense search results: ${denseResults.matches?.length || 0}`);

        // Perform sparse search (only if we have valid sparse vector)
        let sparseResults: any = { matches: [] };
        
        if (sparseVector.indices.length > 0 && sparseVector.values.length > 0) {
            const sparseIndex = type === 'documents' ? docSparseIndex : transcriptSparseIndex;
            const sparseSearchParams: any = {
                topK: topK * 2,
                includeMetadata: true,
                includeValues: false,
                sparseVector: {
                    indices: sparseVector.indices,
                    values: sparseVector.values
                },
                ...(type === 'transcripts' && userId && { filter: { clientId: userId } })
            };
            
            sparseResults = await sparseIndex.query(sparseSearchParams);
        } else {
            console.log(`⚠️ Skipping sparse search - no valid sparse vector generated`);
        }

        console.log(`📈 Sparse search results: ${sparseResults.matches?.length || 0}`);

        // Combine and rank results
        const combinedResults = combineSearchResults(
            denseResults.matches || [],
            sparseResults.matches || [],
            type
        );

        console.log(`🔄 Combined results: ${combinedResults.length}`);
        console.log(`=== END PINECONE HYBRID SEARCH ===\n`);

        return combinedResults.slice(0, topK);

    } catch (error) {
        console.error('Error in Pinecone hybrid search:', error);
        return [];
    }
}

function combineSearchResults(
    denseMatches: PineconeMatch[],
    sparseMatches: PineconeMatch[],
    type: string
): HybridSearchResult[] {
    console.log(`🔀 Combining ${denseMatches.length} dense + ${sparseMatches.length} sparse results`);

    // Create maps for easy lookup
    const denseScores = new Map<string, number>();
    const sparseScores = new Map<string, number>();
    const documentsMap = new Map<string, Document>();

    // Process dense results
    denseMatches.forEach(match => {
        denseScores.set(match.id, match.score);
        if (match.metadata) {
            documentsMap.set(match.id, new Document({
                pageContent: match.metadata.text || '',
                metadata: match.metadata
            }));
        }
    });

    // Process sparse results
    sparseMatches.forEach(match => {
        sparseScores.set(match.id, match.score);
        if (match.metadata && !documentsMap.has(match.id)) {
            documentsMap.set(match.id, new Document({
                pageContent: match.metadata.text || '',
                metadata: match.metadata
            }));
        }
    });

    // Get all unique document IDs
    const allIds = new Set([...denseScores.keys(), ...sparseScores.keys()]);

    // Normalize scores to 0-1 range
    const maxDenseScore = Math.max(...Array.from(denseScores.values()), 0.001);
    const maxSparseScore = Math.max(...Array.from(sparseScores.values()), 0.001);

    // Combine scores for each document
    const hybridResults: HybridSearchResult[] = [];

    for (const id of allIds) {
        const document = documentsMap.get(id);
        if (!document) continue;

        const normalizedDenseScore = (denseScores.get(id) || 0) / maxDenseScore;
        const normalizedSparseScore = (sparseScores.get(id) || 0) / maxSparseScore;

        const combinedScore = 
            (normalizedDenseScore * CONFIG.SEARCH.HYBRID.DENSE_WEIGHT) +
            (normalizedSparseScore * CONFIG.SEARCH.HYBRID.SPARSE_WEIGHT);

        hybridResults.push({
            document,
            denseScore: normalizedDenseScore,
            sparseScore: normalizedSparseScore,
            combinedScore
        });
    }

    // Sort by combined score
    hybridResults.sort((a, b) => b.combinedScore - a.combinedScore);

    console.log(`📊 Top 5 combined scores:`);
    hybridResults.slice(0, 5).forEach((result, index) => {
        console.log(`  ${index + 1}. Dense: ${result.denseScore.toFixed(3)}, Sparse: ${result.sparseScore.toFixed(3)}, Combined: ${result.combinedScore.toFixed(3)}`);
    });

    return hybridResults;
}

// Helper function to perform query-time hybrid search with Pinecone's native hybrid search (if available)
export async function performNativeHybridSearch(
    query: string,
    type: 'documents' | 'transcripts',
    userId?: number,
    topK: number = 10
): Promise<HybridSearchResult[]> {
    console.log(`🚀 Attempting Pinecone native hybrid search...`);

    try {
        // Generate both embeddings
        const denseVector = await embeddings.embedQuery(query);
        const sparseGenerator = type === 'documents' ? documentSparseGenerator : transcriptSparseGenerator;
        const sparseVector = sparseGenerator.generateSparseEmbedding(query);

        // Use the dense index as primary (you can modify this based on your Pinecone setup)
        const primaryIndex = type === 'documents' ? docDenseIndex : transcriptDenseIndex;
        
        const searchParams: any = {
            topK,
            includeMetadata: true,
            vector: denseVector,
            sparseVector: {
                indices: sparseVector.indices,
                values: sparseVector.values
            }
        };

        // Add user filter for transcripts
        if (type === 'transcripts' && userId) {
            searchParams.filter = { clientId: userId };
        }

        const results = await primaryIndex.query(searchParams);

        console.log(`✅ Native hybrid search returned ${results.matches?.length || 0} results`);

        return (results.matches || []).map((match: any) => ({
            document: new Document({
                pageContent: match.metadata?.text || '',
                metadata: match.metadata || {}
            }),
            denseScore: match.score * CONFIG.SEARCH.HYBRID.DENSE_WEIGHT,
            sparseScore: match.score * CONFIG.SEARCH.HYBRID.SPARSE_WEIGHT,
            combinedScore: match.score
        }));

    } catch (error) {
        console.warn(`⚠️ Native hybrid search failed, falling back to separate searches: ${error}`);
        return performPineconeHybridSearch(query, type, userId, topK);
    }
}

export type { HybridSearchResult }; 