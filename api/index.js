import { OpenAI } from 'openai';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
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
        // Explicitly check for incoming payload structures
        if (!request.body || !request.body.messages) {
            return response.status(200).json({
                choices: [{ message: { role: "assistant", content: "⚠️ **Frontend Error:** The message array payload is reaching the backend empty. Please check your text input handler state." } }]
            });
        }

        const { messages } = request.body;

        const completion = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages: messages,
            temperature: 0.5
        });

        return response.status(200).json(completion);

    } catch (error) {
        console.error('Core Server System Error Catch:', error);
        
        // Render exact error text straight to your UI for transparent debugging
        return response.status(200).json({
            choices: [{
                message: {
                    role: "assistant",
                    content: `⚠️ **Backend Core System Exception:**\n\n\`\`\`text\n${error.message || 'Unknown runtime error'}\n\`\`\`\n\nEnsure your OPENAI_API_KEY environment variable is configured in your Vercel project settings tab.`
                }
            }]
        });
    }
}