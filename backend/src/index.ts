import express, { type Request, type Response } from "express";
import { processQuery } from "./main/query";    
import { initializeRAGSystem } from "./config/initialize";
import { ingestDocuments, ingestTranscripts } from './main/ingestion';

const app = express();
app.use(express.json());

async function initialize() {
    try {
        console.log('🚀 Initializing RAG system...');
        await initializeRAGSystem();
        
        // console.log('📚 Starting document ingestion process...');
        // await ingestDocuments();

        // console.log('💬 Starting transcript ingestion process...');
        // await ingestTranscripts();
        // console.log('✅ Ingestion process complete.');
        
        return true;
    } catch (error) {
        console.error('❌ Initialization failed:', error);
        process.exit(1);
    }
}

async function startServer() {
    try {
        console.log('🔧 Starting RAG system initialization...');
        await initialize();
        
        const PORT = 3003;
        app.listen(PORT, () => {
            console.log(`🎉 Server is running on port ${PORT}`);
            console.log('📋 RAG system is ready to accept queries!');
        });
    } catch (error) {
        console.error('❌ Failed to start server:', error);
        process.exit(1);
    }
}

app.get("/", (_, res: Response) => {
    res.json({ message: "RAG System is running" });
});

// interface QuestionRequest {
//     question: string;
//     clientId?: number; // 1 = nathan, 2 = robert
// }

app.post("/ask", async (req: Request, res: Response) => {
    try {
        const { question, clientId } = req.body;

        if (!question) {
            res.status(400).json({ 
                error: "Question is required" 
            });
            return;
        }
        if(!clientId) {
            res.status(400).json({ 
                error: "Client ID is required" 
            });
            return;
        }

        const userId = clientId == 1 ? 'nathan' : 'robert'

        const result = await processQuery(question, userId);

        res.json({
            answer: result.answer,
            metadata: {
                confidence: result.retrievalResult.confidence,
                sources: result.retrievalResult.sources,
                strategy: result.retrievalResult.searchStrategy
            }
        });
    } catch (error) {
        console.error('Error processing question:', error);
        res.status(500).json({ 
            error: "Failed to process question",
            details: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});

// Start the server with initialization
startServer();

export default app;