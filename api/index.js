import { OpenAI } from 'openai';

// Initialize the OpenAI SDK client to act as a proxy routing to GitHub's inference server
const openai = new OpenAI({
    baseURL: "https://models.github.ai/inference",
    apiKey: process.env.GITHUB_TOKEN
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

        // Call the GitHub Marketplace model engine
        const completion = await openai.chat.completions.create({
            model: 'GPT-4.1-mini', 
            messages: messages,
            temperature: 0.5
        });

        return response.status(200).json(completion);

    } catch (error) {
        console.error('GitHub Models Pipeline Failure Exception:', error);
        
        // Output clear diagnostics straight onto your UI chat screen for faster debugging
        return response.status(200).json({
            choices: [{
                message: {
                    role: "assistant",
                    content: `⚠️ **GitHub Models API Gateway Exception:**\n\n\`\`\`text\n${error.message || 'Unknown runtime error'}\n\`\`\`\n\nPlease ensure your GITHUB_TOKEN environment variable is saved securely inside your Vercel settings tab.`
                }
            }]
        });
    }
}