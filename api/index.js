module.exports = async (req, res) => {
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
            return res.status(500).json({ error: "Server error: GITHUB_TOKEN environment variable is unconfigured." });
        }

        const sanitizedMessages = messages.map(msg => ({
            role: msg.role === "assistant" || msg.role === "user" || msg.role === "system" ? msg.role : "user",
            content: String(msg.content)
        }));

        const response = await fetch("https://models.github.ai/inference/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                messages: sanitizedMessages,
                model: "openai/gpt-4.1",
                temperature: 0.7,
                max_tokens: 512
            })
        });

        if (!response.ok) {
            const errorReason = await response.text();
            console.error(`GitHub Models Endpoint rejected request [${response.status}]:`, errorReason);
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