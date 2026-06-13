import { OpenAI } from 'openai';

export const config = {
    maxDuration: 20, 
};

const openai = new OpenAI({
    baseURL: "https://models.github.ai/inference",
    apiKey: process.env.GITHUB_TOKEN
});

export default async function handler(request, response) {
    // Standard Cross-Origin Headers
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
            return response.status(400).json({ error: "Missing message context payload." });
        }

        const { messages } = request.body;

        // --- ATTEMPT 1: Try Primary GPT-4.1-mini with Native Abort Signal ---
        try {
            console.log("Routing query to primary model: GPT-4.1-mini");
            
            // Generate a real network abort signal
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 7500); // Strict 7.5 second network cutoff

            const completion = await openai.chat.completions.create({
                model: 'GPT-4.1-mini', 
                messages: messages,
                temperature: 0.3
            }, {
                signal: controller.signal // Enforces hardware level request termination
            });
            
            clearTimeout(timeoutId); // Clear timeout if model responds fast
            return response.status(200).json(completion);

        } catch (primaryError) {
            console.warn("Primary model stalled or timed out. Activating Gemini 2.5 Flash-Lite failover...", primaryError.message);

            // --- ATTEMPT 2: Failover Backup to Gemini 2.5 Flash-Lite ---
            const backupCompletion = await openai.chat.completions.create({
                model: 'google/gemini-2.5-flash-lite', 
                messages: messages,
                temperature: 0.3
            });

            console.log("Successfully served payload via Gemini 2.5 Flash-Lite fallback pipeline.");
            return response.status(200).json(backupCompletion);
        }

    } catch (globalError) {
        console.error('Complete routing fallback collapse:', globalError);
        
        return response.status(200).json({
            choices: [{
                message: {
                    role: "assistant",
                    content: `⚠️ **Inference Gateway Error:** Both primary and fallback models are currently unresponsive. Let's stay connected directly via [LinkedIn](https://linkedin.com/in/vishalkhandatascience) or email me at vishalkhan251@gmail.com!`
                }
            }]
        });
    }
}