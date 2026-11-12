# Financial RAG Assistant 📈🤖

Welcome! This application is an AI-powered Retrieval-Augmented Generation (RAG) assistant designed to analyze and answer questions based on **NVIDIA Corporation's SEC Filings (Form 10-Q / 10-K)** and financial reports.

---

## 💡 What You Can Ask
- **Financial Performance:** "What was NVIDIA's Data Center revenue in Q2 Fiscal 2027?"
- **Segment Breakdown:** "Compare the revenue growth between Compute & Networking and Graphics."
- **Capital Returns & Debt:** "How much did NVIDIA spend on share repurchases in Q2?"
- **Risk Factors & Supply Chain:** "What does the 10-Q filing state about Blackwell system supply constraints and export controls?"

---

## ⚡ Features
- **Grounded Financial QA:** Answers are generated strictly from ingested SEC document chunks using Qdrant vector retrieval.
- **Multimodal Voice Input & Output:** Use the voice button to speak your questions (Groq Whisper STT) and listen to synthesized answers (ElevenLabs TTS).