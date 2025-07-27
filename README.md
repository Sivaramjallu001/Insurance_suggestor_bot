# 🛡️ Insurance Q&A Bot with Language Translation & Memory

A multilingual, intelligent Insurance Q&A chatbot built using **Streamlit**, **Google Gemini**, **SentenceTransformers**, and **Deep Translator**. This app helps users understand various insurance policy documents by answering their questions clearly, based only on relevant policies and user context. It supports multiple Indian languages and remembers past queries for continuity.

---

## 📌 Features

- 🔍 **Semantic Policy Matching**: Automatically selects the top relevant insurance policies based on user queries using Sentence Transformers.
- 🧠 **Contextual Memory**: Maintains a short-term memory of the last 10 questions and answers to support contextual conversations.
- 🌐 **Multilingual Support**: Answers can be translated into 10+ Indian languages including Hindi, Telugu, Tamil, and Bengali.
- 📚 **Policy-Specific Answers**: Gemini AI generates answers **strictly based** on selected policy texts to ensure accuracy.
- 🔁 **Explore More**: Allows switching to different policies if the user wants to explore other options for the same question.
- 🧹 **Memory Reset**: Option to clear memory and start a new inquiry anytime.

---

## 🏗️ Tech Stack

- [Streamlit](https://streamlit.io/) – Web app framework
- [Google Gemini API](https://ai.google.dev/) – LLM used for Q&A generation
- [SentenceTransformers](https://www.sbert.net/) – Embedding model for semantic search (`all-MiniLM-L6-v2`)
- [Torch](https://pytorch.org/) – Used for cosine similarity calculations
- [Deep Translator](https://pypi.org/project/deep-translator/) – Language translation
- Python `collections.deque` – Used for storing short-term memory (up to 10 Q&As)

---

## 🚀 How It Works

1. **User Asks a Question**
   - The app uses semantic search to identify the top 3 most relevant policy documents.
2. **Gemini is Prompted**
   - Gemini is given:
     - User’s previous context (last 10 Q&As)
     - Selected policy text
     - Current question
   - It generates an accurate, policy-specific response.
3. **Answer is Translated**
   - Response is optionally translated into the selected Indian language.
4. **Answer is Displayed & Stored**
   - The answer is shown and stored in memory for future reference.

---

## 📁 Folder Structure

```bash
project/
│
├── app.py                      # Main Streamlit app
├── policies/                   # Folder containing policy_1_text to policy_200_text
├── requirements.txt            # Python dependencies
└── README.md                   # You're reading it!
