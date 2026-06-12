module.exports = async (req, res) => {
    // 1. Establish robust CORS handling parameters for GitHub Pages asset origins
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
            return res.status(500).json({ error: "Server error: GITHUB_TOKEN variable is unconfigured." });
        }

        // Deep sanitize request context payloads to strip non-standard nested keys
        const sanitizedMessages = messages.map(msg => ({
            role: msg.role === "assistant" || msg.role === "user" || msg.role === "system" ? msg.role : "user",
            content: String(msg.content)
        }));

        // 2. Query target catalog using standard string paths
        const response = await fetch("https://models.inference.ai.azure.com/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                messages: sanitizedMessages,
                model: "openai/gpt-4.1", // Realigned directly to the GPT-4.1 engine marketplace name
                temperature: 0.7,
                max_tokens: 512
            })
        });

        // Catch and output text details if the upstream gateway blocks our call
        if (!response.ok) {
            const errorReason = await response.text();
            console.error(`Upstream Catalog API returned error status [${response.status}]:`, errorReason);
            return res.status(response.status).json({ 
                error: `Upstream error status: ${response.status}`, 
                details: errorReason 
            });
        }

        const data = await response.json();
        return res.status(200).json(data);

    } catch (error) {
        console.error("Critical Exception in Vercel Serverless Function:", error);
        return res.status(500).json({ error: "Internal Server Error", details: error.message });
    }
};