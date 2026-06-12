module.exports = async (req, res) => {
    // 1. Handle CORS Pre-flight headers safely
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
        return res.status(200).end();
    }

    if (req.method !== "POST") {
        return res.status(405).json({ error: "Method Not Allowed" });
    }

    try {
        const { messages } = req.body;
        const token = process.env.GITHUB_TOKEN;

        if (!token) {
            return res.status(401).json({ error: "Server Configuration Error: Missing GITHUB_TOKEN" });
        }

        // Sanitize the messages array to ensure NO hidden metadata attributes leak through
        const sanitizedMessages = messages.map(msg => ({
            role: msg.role,
            content: msg.content
        }));

        // 2. HTTP POST fetch request directly to GitHub AI Models
        const response = await fetch("https://models.inference.ai.azure.com/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                messages: sanitizedMessages,
                model: "meta-llama-3-8b-instruct",
                temperature: 0.7,
                max_tokens: 512
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Upstream API Error Response Status:", response.status, errorText);
            // Return the specific message back to help us debug exactly what it doesn't like
            return res.status(400).json({ error: errorText });
        }

        const data = await response.json();
        return res.status(200).json(data);

    } catch (error) {
        console.error("Vercel Serverless Function Exception Error:", error);
        return res.status(500).json({ error: error.message });
    }
};