const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");
const path = require("path");

const app = express();
const PORT = 3000;

// 🔑 Replace with your actual Gemini API key
require("dotenv").config({ path: path.join(__dirname, "private", ".env") });
const API_KEY = process.env.GEMINI_API_KEY;

// 📦 Middleware to parse incoming JSON and serve static files (like HTML)
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname,"public")));
var first=1;
// 🚀 Handle chat messages
app.post("/chat", async (req, res) => {
  userMessage = req.body.message;
  if(first==1){
    const extra="Please do not use markdown or any formatting that cannot be rendered in a text environment. Do not use asterisks as delineators. Use newline chars instead. Use appropriate punctuation. ";
    userMessage=extra + userMessage;
    // console.log(userMessage);
    first=0;
  }
  try {
    // 📤 Send message to Gemini via REST API
    const response = await axios.post(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${API_KEY}`,
      {
        contents: [{ role: "user", parts: [{ text: userMessage }] }],
      }
    );

    // 📥 Extract Gemini's response
    const reply = response.data.candidates?.[0]?.content?.parts?.[0]?.text || "No response";
    res.json({ reply });
  } catch (error) {
    console.error("Gemini error:", error.response?.data || error.message);
    res.status(500).json({ reply: "Error communicating with Gemini API" });
  }
});

// 🌐 Start the server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});