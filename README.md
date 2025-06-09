# 🤖 PMAY Chatbot - MoHUA RAG-based Assistant

A sophisticated chatbot built for the Ministry of Housing and Urban Affairs (MoHUA) to assist users with queries related to the Pradhan Mantri Awas Yojana (PMAY) scheme. This application uses advanced RAG (Retrieval Augmented Generation) with cross-encoder re-ranking to provide accurate and context-aware responses.

## 🌟 Features

- **Intelligent Document Processing**: Upload and process PDF documents containing PMAY-related information
- **Advanced RAG Implementation**: Uses ChromaDB for vector storage and retrieval
- **Cross-Encoder Re-ranking**: Improves response relevance using semantic re-ranking
- **Streamlit Interface**: User-friendly chat interface with real-time responses
- **Local LLM Integration**: Powered by Ollama for privacy and offline capabilities
- **Context-Aware Responses**: Provides accurate information based on official documents

## 🚨 Prerequisites

- Python > 3.10
- SQLite > 3.35
- [Ollama](https://ollama.dev/download) installed and running locally
- Sufficient disk space for model downloads

## 🔧 Setup Instructions

1. **Clone the repository**
   ```sh
   git clone <repository-url>
   cd pmay-chatbot-llm-rag
   ```

2. **Set up the environment**
   ```sh
   make setup
   ```
   This will:
   - Create a Python virtual environment
   - Install all required dependencies
   - Set up the development environment

3. **Install required models**
   ```sh
   make install-models
   ```
   This will download the necessary Ollama models:
   - `nomic-embed-text:latest` for embeddings
   - `llama3.2:1b` for chat completions

4. **Verify Ollama is running**
   ```sh
   make check-ollama
   ```
   This ensures Ollama is running properly on port 11434

5. **Run the application**
   ```sh
   make run
   ```
   This will start the Streamlit interface

## 📚 Usage

1. **Upload Documents**
   - Upload PDF documents containing PMAY-related information
   - The system will automatically process and index the content

2. **Ask Questions**
   - Use the chat interface to ask questions about PMAY
   - The chatbot will provide responses based on the uploaded documents
   - View source documents for each response using the expandable section

## 🛠️ Development

### Available Commands

```sh
make help  # Show all available commands
```

Common development commands:
```sh
make check     # Run linting checks
make fix       # Fix auto-fixable linting issues
make clean     # Clean temporary files and caches
make run       # Run the application
```

### Project Structure

- `app.py`: Main application file containing the Streamlit interface and core logic
- `models/`: Directory for storing downloaded models
- `demo-rag-chroma/`: Vector database storage directory
- `requirements/`: Project dependencies
  - `requirements.txt`: Main project dependencies
  - `requirements-dev.txt`: Development dependencies

## 🔍 Technical Details

- **Vector Database**: ChromaDB with Ollama embeddings
- **Text Splitting**: RecursiveCharacterTextSplitter with 750 token chunks
- **Re-ranking**: Cross-encoder model (ms-marco-MiniLM-L-6-v2)
- **LLM**: Local Llama model through Ollama
- **UI Framework**: Streamlit with custom CSS styling

## ⚠️ Common Issues and Solutions

1. **ChromaDB/SQLite Compatibility**
   - If you encounter SQLite-related errors, refer to [ChromaDB troubleshooting](https://docs.trychroma.com/troubleshooting#sqlite)

2. **Model Download Issues**
   - Ensure stable internet connection for initial model downloads
   - Check available disk space in the `models/` directory
   - Run `make install-models` to retry model downloads

3. **Ollama Connection**
   - Verify Ollama is running on port 11434 using `make check-ollama`
   - Check if required models are downloaded in Ollama
   - Restart Ollama if connection issues persist

4. **Environment Issues**
   - If you encounter dependency issues, try:
     ```sh
     make clean
     make setup
     ```
   - Ensure you're using Python 3.10 or higher

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add appropriate license information]

## 🔗 Useful Links

- [PMAY Official Website](https://pmay-urban.gov.in/)
- [MoHUA Official Website](https://mohua.gov.in/)
- [Ollama Documentation](https://ollama.ai/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)

# 📑 RAG with Cross-Encoders Re-ranking Demo Application

Demo LLM app with RAG for the YouTube video.

🚨 NOTE: **Requires `Python > 3.10` with  `SQLite > 3.35`**

Watch the video 👇

<a href="https://youtu.be/1y2TohQdNbo">
<img src="https://i.imgur.com/lJUyEm1.png" width="800">
</a>

---

### Part II: Watch Optimizing RAG with Semantic Cache ⚡️

<a href="https://youtu.be/0agBo7_wKTo">
<img src="https://i.imgur.com/rJJaHmG.png" width="400">
</a>

🐙 [**Codebase**](https://github.com/yankeexe/llm-rag-with-reranker-demo/tree/semantic-cache)


## 🤸‍♀️ Getting Help

```sh
make

# OR

make help
```
