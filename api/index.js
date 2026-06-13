import { OpenAI } from 'openai';

export const config = {
    maxDuration: 20, 
};

// Access the GitHub Models endpoint via your existing credentials
const openai = new OpenAI({
    baseURL: "https://models.github.ai/inference",
    apiKey: process.env.GITHUB_TOKEN,
    timeout: 8500 // Cut the primary model off at 8.5 seconds to leave room for the backup
});

export default async function handler(request, response) {
    // Inject vital response cross-origin headers
    response.setHeader('Access-Control-Allow-Credentials', true);
    response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
    response.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

    if (request.method === 'OPTIONS') {
        return response.status(200).end();
    }

    if (request.method !== 'POST') {
        return response.status(405).json({ error: 'POST requests allowed only.' });
    }

    try {
        if (!request.body || !request.body.messages) {
            return response.status(400).json({ error: "Missing payload chat history message array." });
        }

        const { messages } = request.body;

        // --- ATTEMPT 1: Primary Model Execution ---
        try {
            console.log("Routing query to primary model: GPT-4.1-mini");
            const completion = await openai.chat.completions.create({
                model: 'GPT-4.1-mini', 
                messages: messages,
                temperature: 0.3
            });
            
            return response.status(200).json(completion);

        } catch (primaryError) {
            console.warn("Primary model failed or timed out. Swapping to backup engine...", primaryError.message);

            // --- ATTEMPT 2: Failover Backup Execution ---
            // Gemini 2.5 Flash-Lite is optimized for ultra-low latency, making it the perfect backup
            const backupCompletion = await openai.chat.completions.create({
                model: 'gemini-2.5-flash-lite', 
                messages: messages,
                temperature: 0.3,
                timeout: 8000 // Give the backup its own dedicated execution buffer
            });

            console.log("Successfully served payload via Gemini 2.5 Flash-Lite failover pipeline.");
            return response.status(200).json(backupCompletion);
        }

    } catch (globalError) {
        console.error('Complete pipeline exhaustion:', globalError);
        
        // Final ultimate defense notification to prevent frontend loops if both systems drop
        return response.status(200).json({
            choices: [{
                message: {
                    role: "assistant",
                    content: `⚠️ **Inference Engine Network Alert:** Both primary and backup models on the marketplace are taking too long to reply right now. Let's stay connected directly via [LinkedIn](https://linkedin.com/in/vishalkhandatascience) or email me at vishalkhan251@gmail.com!`
                }
            }]
        });
    }
}