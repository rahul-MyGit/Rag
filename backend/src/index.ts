import express, { type Request, type Response } from "express";
import { processQuery } from "./main/query";    
import { initializeRAGSystem } from "./config/initialize";
import { ingestDocuments, ingestTranscripts } from './main/ingestion';

const app = express();
app.use(express.json());

async function initialize() {
    try {
        console.log('Initializing RAG system...');
        await initializeRAGSystem();
        
        // console.log('Starting ingestion process...');
        // await ingestDocuments();

        // console.log('Starting transcript ingestion process...');
        // await ingestTranscripts();
        console.log('Ingestion process complete.');
    } catch (error) {
        console.error('Initialization failed:', error);
        process.exit(1);
    }
}

// Start initialization
initialize();

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

        // Map clientId to userId
        const userId = clientId == 1 ? 'nathan' : 
                      clientId == 2 ? 'robert' : 
                      undefined;

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

app.listen(3000, () => {
    console.log("Server is running on port 3000");
});

export default app;