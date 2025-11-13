A chatbot where users can upload multiple PDFs and ask questions about their contents — all within an intuitive web interface.

This system uses:
- Sentence Transformers (all-MiniLM-L6-v2) for embeddings
- Hugging Face LLM (Llama-3.1-8B-Instruct) for contextual and intelligent answers
- LangChain for retrieval orchestration and FAISS for efficient vector search
- Streamlit for a fast and interactive user experience
It performs semantic retrieval using FAISS vector storage and maintains conversational memory for smooth multi-turn dialogue.


⚙️ Setup
- Create a virtual environment
- Install dependencies
- Create a .env file
- Run the Streamlit app

🧩 How It Works

- PDF Upload & Parsing

  Extracts text from each uploaded PDF using PyPDF2.

- Text Chunking

  Splits text into overlapping chunks (default: 1000 characters, 200 overlap) for contextual retrieval.

- Embedding Generation

  Encodes text chunks into vector embeddings using the free model sentence-transformers/all-MiniLM-L6-v2.

- Vector Store Creation (FAISS)

  All embeddings are stored in a FAISS index for efficient semantic similarity search.

- Conversational Chain

  When a user asks a question, the app retrieves the most relevant chunks, sends them with the query to the LLM (meta-llama/Llama-3.1-8B-Instruct via Hugging Face), and returns an answer.

- Memory & Context

  The ConversationBufferMemory keeps prior chat history for coherent follow-up questions.
