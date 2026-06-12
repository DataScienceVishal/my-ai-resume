module.exports = async (req, res) => {
    // 1. Establish strict CORS headers for your client-side assets
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

        // Stripping array payloads down to exact matching data schemas
        const sanitizedMessages = messages.map(msg => ({
            role: msg.role === "assistant" || msg.role === "user" || msg.role === "system" ? msg.role : "user",
            content: String(msg.content)
        }));

        // 2. Transmit standardized request body payload to upstream Azure/GitHub API
        const response = await fetch("https://models.inference.ai.azure.com/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                messages: sanitizedMessages,
                model: "Meta-Llama-3-8B-Instruct", // Re-aligned to exact marketplace nomenclature
                temperature: 0.7,
                max_tokens: 512
            })
        });

        // 3. Robust Error Logging: If it's a 400, catch the exact message string
        if (!response.ok) {
            const errorReason = await response.text();
            console.error(`Upstream Catalog Rejected Payload with status [${response.status}]:`, errorReason);
            
            // Pass the exact upstream reason straight back to your browser console to read it clearly
            return res.status(response.status).json({ 
                error: `Upstream error: ${response.status}`, 
                details: errorReason 
            });
        }

        const data = await response.json();
        return res.status(200).json(data);

    } catch (error) {
        console.error("Vercel Function Critical Exception Runtime Crash:", error);
        return res.status(500).json({ error: "Internal Server Exception", details: error.message });
    }
};