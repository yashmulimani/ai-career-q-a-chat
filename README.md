Hybrid RAG Chatbot for Career Guidance

An AI-powered career guidance chatbot that helps users explore technology careers, learning roadmaps, interview preparation, and technical knowledge.
The system implements a Hybrid Retrieval-Augmented Generation (RAG) architecture combining:

🔎 Vector search (semantic retrieval)

🧠 BM25 keyword search

🌐 Web search fallback

🤖 Large Language Model (Mistral-7B)

This allows the chatbot to provide accurate, context-grounded, and up-to-date responses.

🚀 Demo
Example query:
Give me the roadmap for becoming an AI/ML engineer

Response includes:
- Programming foundations
- Machine learning concepts
- Deep learning frameworks
- Cloud & MLOps tools
- Projects and career guidance

✨ Features
🤖 AI Career Guidance
Ask questions about:
- Career paths in technology
- Required skills for roles
- Career transitions
- Industry trends
- Example:
- Is AI a good career in 2026?
- 
🗺 Career Roadmaps
The chatbot provides structured roadmaps for roles like:

- AI / ML Engineer
- Cloud Engineer
- DevOps Engineer
- Backend Developer
- Data Engineer
- Mobile Developer

Example:
Give me a roadmap for becoming a Cloud Engineer
📚 Knowledge-Grounded Answers

The chatbot retrieves information from curated documents stored in:
career_docs/
faq_docs/
tech_docs/
Interview_qa_docs/
roadmap_docs/

These include:
- career explanations
- technology knowledge
- FAQs
- interview preparation
- learning roadmaps

🔍 Hybrid Retrieval System

The chatbot combines two retrieval methods:

1️⃣ Semantic Search
Uses sentence-transformer embeddings stored in a Chroma vector database.
This allows the model to understand meaning, not just keywords.

Example:
DevOps roadmap
will match:
DevOps learning path

2️⃣ Keyword Search (BM25)
BM25 improves recall by matching exact keywords.
This ensures the chatbot retrieves the most relevant documents.

🌐 Web Search Fallback
If the knowledge base lacks sufficient information, the chatbot performs DuckDuckGo web search to retrieve external information.
This allows the assistant to answer questions about recent trends and new technologies.

💬 Interactive Chat UI
The application uses Streamlit to create a clean and interactive chat interface.
Users can interact with the AI just like a conversational assistant.

🧠 System Architecture
User Query
     │
     ▼
Hybrid Retrieval
(Vector Search + BM25 Keyword Search)
     │
     ▼
Relevant Knowledge Documents
     │
     ▼
Web Search (if context insufficient)
     │
     ▼
Prompt Construction
     │
     ▼
Mistral-7B Large Language Model
     │
     ▼
AI Response

⚙️ Tech Stack
Component	               Technology
Frontend	               Streamlit
LLM	                    Mistral-7B (HuggingFace)
Embeddings	          sentence-transformers
Vector Database	     Chroma
Keyword Search	          BM25
Web Search	          DuckDuckGo
Framework	               LangChain
Language	               Python

📂 Project Structure
AI-Career-Coach
│
├── app.py                # Streamlit chatbot application
├── rag_chat.py           # CLI chatbot version
├── llm.py                # LLM configuration
├── requirements.txt      # Project dependencies
│
├── career_docs/          # Career information
├── faq_docs/             # Frequently asked questions
├── tech_docs/            # Technology knowledge
├── Interview_qa_docs/    # Interview preparation
├── roadmap_docs/         # Career roadmaps
│
└── .env                  # API keys (ignored)
