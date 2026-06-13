// api/chat.js
import { OpenAI } from "openai";

// Initialize the OpenAI SDK utilizing Vercel's secure environment variables
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export default async function handler(req, res) {
  // Enforce global CORS and Preflight handling options
  res.setHeader("Access-Control-Allow-Credentials", true);
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS,PATCH,DELETE,POST,PUT");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version"
  );

  // Instantly resolve browser OPTIONS preflight checks
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed. Use POST." });
  }

  try {
    const { messages } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "Invalid context payload history." });
    }

    // Forward the chat history straight to OpenAI's production model endpoint
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini", // Cost-efficient, high-speed model optimal for resumes
      messages: messages,
      temperature: 0.7,
    });

    // Return a structured JSON response matching your frontend layout reader
    return res.status(200).json(completion);

  } catch (error) {
    console.error("OpenAI Gateway Internal Failure:", error);

    // If OpenAI directly returns a 429 rate-limit error, catch it and explain why
    if (error.status === 429) {
      return res.status(429).json({
        error: "OpenAI API Quota Exceeded. Please check your developer billing dashboard balance.",
      });
    }

    return res.status(500).json({ error: error.message || "Internal server pipeline breakdown." });
  }
}