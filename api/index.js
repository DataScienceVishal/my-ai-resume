import { OpenAI } from 'openai';

// Force Vercel to allow the maximum execution window possible for Hobby accounts
export const config = {
    maxDuration: 60, 
};

// Initialize the OpenAI SDK client with a firm connection timeout
const openai = new OpenAI({
    baseURL: "https://models.github.ai/inference",
    apiKey: process.env.GITHUB_TOKEN,
    timeout: 9000 // 9-second timeout limit to stay safely under Vercel's 10s drop limit
});

export default async function handler(request, response) {
    // Cross-Origin Resource Sharing (CORS) Headers
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

        // Call the GitHub Marketplace model engine with a slightly lower temperature for faster responses
        const completion = await openai.chat.completions.create({
            model: 'GPT-4.1-mini', 
            messages: messages,
            temperature: 0.3
        });

        return response.status(200).json(completion);

    } catch (error) {
        console.error('GitHub Models Pipeline Failure Exception:', error);
        
        // Output clean diagnostic message straight to the chat UI if the endpoint times out
        return response.status(200).json({
            choices: [{
                message: {
                    role: "assistant",
                    content: `⚠️ **Connection Timeout:** The GitHub inference server is taking unusually long to respond right now. Please try asking your question again in a brief moment, or let's connect directly via [LinkedIn](https://linkedin.com/in/vishalkhandatascience)!`
                }
            }]
        });
    }
}