import { llm } from "../config/initialize";
import { type RetrievalResult } from "../types";

export async function generateResponse(query: string, retrievalResult: RetrievalResult): Promise<string> {
    if (!retrievalResult.content) {
      return "I couldn't find relevant information to answer your question. Please try rephrasing your question or providing more specific details.";
    }
    
    const sources = retrievalResult.sources.map(s => s.id).join(', ');
    
    const prompt = `You are a helpful assistant that answers questions based on provided context. Use the following context to answer the user's question comprehensively and accurately.
  
  Context:
  ${retrievalResult.content}
  
  Question: ${query}
  
  Instructions:
  - Provide a detailed and well-structured answer based on the context
  - If information comes from transcripts, mention the speaker when relevant
  - If the context doesn't fully answer the question, clearly state what information is missing
  - Use specific examples and details from the context when possible
  - Maintain a professional and helpful tone
  - If there are conflicting information sources, acknowledge and explain the differences
  
  Answer:`;
  
    try {
      const response = await llm.invoke(prompt);
      const answer = response.content as string;
      
      const confidenceNote = retrievalResult.confidence < 0.7 
        ? '\n\n*Note: This answer is based on potentially incomplete information.*'
        : '';
      
      return `${answer}${confidenceNote}\n\n**Sources:** ${sources}`;
    } catch (error) {
      console.error('Error generating response:', error);
      return 'Sorry, I encountered an error while generating the response. Please try again.';
    }
}