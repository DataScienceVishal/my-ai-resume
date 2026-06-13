import { OpenAI } from 'openai';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

export default async function handler(req, res) {
    // Inject vital response headers to satisfy browser cross-origin requests
    res.setHeader('Access-Control-Allow-Credentials', true);
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
    res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

    if (req.method === 'OPTIONS') {
        return res.status(200).end();
    }

    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method tracking restriction active. Use POST requests.' });
    }

    try {
        const { messages } = req.body;

        if (!messages || !Array.isArray(messages)) {
            return res.status(400).json({ error: 'Missing or corrupt payload chat history message array.' });
        }

        // Submit complete contextual history payload directly to OpenAI execution core
        const response = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages: messages,
            temperature: 0.5
        });

        // Safeguard response extraction structure
        if (response && response.choices && response.choices[0]) {
            return res.status(200).json(response);
        } else {
            return res.status(502).json({ error: 'Invalid payload response structure returned from OpenAI core.' });
        }

    } catch (error) {
        console.error('Critical Production Gateway Exception:', error);
        
        // Return a verbose JSON payload detailing the exact failure point to your UI
        return res.status(200).json({
            choices: [{
                message: {
                    role: "assistant",
                    content: `⚠️ **Server API Handler Crash Details:**\n\n\`\`\`text\n${error.message || 'Unknown backend error runtime exception'}\n\`\`\`\n\nPlease ensure your OpenAI API Key is valid and your billing limit has not been reached.`
                }
            }]
        });
    }
}