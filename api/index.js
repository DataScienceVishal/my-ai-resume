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

        // 2. Make a clean, native HTTP fetch request directly to the GitHub AI model endpoint
        // This avoids importing any third-party SDKs that trigger 'eval' or CSP violations.
        const response = await fetch("https://models.inference.ai.azure.com/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                messages: messages,
                model: "meta-llama-3-8b-instruct",
                temperature: 0.7,
                max_tokens: 512
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Upstream Model API Error response:", errorText);
            return res.status(response.status).json({ error: `Upstream model error: ${errorText}` });
        }

        const data = await response.json();
        
        // 3. Send the clean JSON payload back to your frontend
        return res.status(200).json(data);

    } catch (error) {
        console.error("Vercel Serverless Function Error:", error);
        return res.status(500).json({ error: error.message });
    }
};