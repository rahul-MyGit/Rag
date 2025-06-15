import express from 'express';
import cors from 'cors';
import { initializeRAGSystem, agencyRouter } from './config/initialize';
import { processQuery } from './main/query';
import type { ChatRequest } from './types';
import type { Request, Response } from 'express';

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

app.get('/health', (_, res: Response) => {
    res.json({ 
        status: 'healthy', 
        system: 'LlamaIndex RAG',
        timestamp: new Date().toISOString()
    });
});

app.post('/ask', async (req: Request, res: Response) => {
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

        res.writeHead(200, {
            'Content-Type': 'text/plain; charset=utf-8',
            'Transfer-Encoding': 'chunked',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        });

        const result = await processQuery(request, agencyRouter);

        const response = result.response;
        const chunkSize = 10; // characters per chunk
        
        for (let i = 0; i < response.length; i += chunkSize) {
            const chunk = response.slice(i, i + chunkSize);
            res.write(chunk);
            await new Promise(resolve => setTimeout(resolve, 50));
        }

        const sources = result.sourceNodes ? result.sourceNodes.map(nodeWithScore => ({
            fileName: nodeWithScore.node.metadata?.fileName || nodeWithScore.node.metadata?.source || 'Unknown',
            score: nodeWithScore.score || 0,
            type: nodeWithScore.node.metadata?.type || 'document'
        })) : [];

        res.write('\n\n---SOURCES---\n');
        res.write(JSON.stringify({ sources, clientId, timestamp: new Date().toISOString() }));
        res.end();

    } catch (error) {
        console.error('Chat endpoint error:', error);
        res.status(500).json({
            error: 'An error occurred while processing your request',
            timestamp: new Date().toISOString()
        });
    }
});

async function startServer() {
    try {
        console.log('Starting RAG system initialization...');
        await initializeRAGSystem();

    } catch (error) {
        console.error('Failed to start server:', error);
        process.exit(1);
    }
}

startServer();
        
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});