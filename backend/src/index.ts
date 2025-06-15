import express from 'express';
import cors from 'cors';
import { initializeRAGSystem, agencyRouter } from './config/initialize';
import { processQuery } from './main/query';
import type { ChatRequest } from './types';

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        system: 'LlamaIndex RAG',
        timestamp: new Date().toISOString()
    });
});

// Main chat endpoint with JSON response
app.post('/ask', async (req, res) => {
    try {
        const request: ChatRequest = req.body;
        const { query, clientId } = request;

        if (!query) {
            res.status(400).json({ error: 'Query is required' });
            return;
        }

        if (!agencyRouter) {
            res.status(500).json({ error: 'RAG system not initialized' });
            return;
        }

        console.log(`🔍 Chat request - Query: "${query}", Client: ${clientId || 'general'}`);

        // Process query
        const result = await processQuery(request, agencyRouter);

        // Extract source information
        const sources = result.sourceNodes ? result.sourceNodes.map(nodeWithScore => ({
            fileName: nodeWithScore.node.metadata?.fileName || nodeWithScore.node.metadata?.source || 'Unknown',
            score: nodeWithScore.score || 0,
            type: nodeWithScore.node.metadata?.type || 'document'
        })) : [];

        // Return JSON response
        res.json({
            response: result.response,
            sources,
            clientId,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Chat endpoint error:', error);
        res.status(500).json({
            error: 'An error occurred while processing your request',
            timestamp: new Date().toISOString()
        });
    }
});

// Initialize and start server
async function startServer() {
    try {
        console.log('🔧 Starting RAG system initialization...');
        await initializeRAGSystem();

    } catch (error) {
        console.error('❌ Failed to start server:', error);
        process.exit(1);
    }
}

// startServer();
        
app.listen(PORT, () => {
    console.log(`🚀 Server is running on port ${PORT}`);
});