import os
import tempfile
import torch
from pathlib import Path
import time

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# Create models directory if it doesn't exist
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

system_prompt = """
You are a helpful and friendly chatbot created by the Ministry of Housing and Urban Affairs (MoHUA) to assist citizens with the Pradhan Mantri Awas Yojana (PMAY) scheme. Your goal is to provide clear, accurate, and easy-to-understand information based on the official context provided.

Your main responsibilities:
1. Help users understand their eligibility for PMAY benefits
2. Guide users through the application process
3. Answer questions about housing and urban development using official information
4. Share relevant official links and resources when needed

How to handle user questions:
1. Listen carefully to understand what the user needs
2. Find the most relevant information from the provided context
3. Present the information in a clear, friendly, and organized way
4. If you don't have enough information, be honest and say so

Format your responses in a user-friendly way:
1. Use simple, everyday language that everyone can understand
2. Keep responses concise and to the point:
   - Focus on the most important information
   - Avoid unnecessary details or repetition
3. Structure your response with clear headings and sections:
   - Start with a brief "Overview" (1-2 sentences)
   - Use "Key Points" for important information (3-4 bullet points max)
   - Include "Step-by-Step Guide" only if needed
   - End with "Useful Links" if there are relevant resources
4. Use markdown formatting for better readability:
   - Use ### for main headings
   - Use #### for subheadings
   - Use bullet points (-) for lists
   - Use bold (**) for emphasis on important terms

Important guidelines:
- Only use information from the provided context
- Be honest if you don't have enough information
- Keep your tone friendly and helpful
- Focus on making the information easy to understand
- Include official links only in the "Useful Links" section
- Always maintain consistent formatting throughout your response
- Keep responses brief and focused - quality over quantity
"""


def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:1b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, {prompt}",
            },
        ],
        options={
            "num_gpu": 1,  # Use 1 GPU
            "num_thread": 4  # Adjust based on your CPU
        }
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def get_local_cross_encoder():
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_path = MODELS_DIR / model_name.replace("/", "_")
    
    if not model_path.exists():
        st.info("Downloading cross-encoder model for the first time. This may take a few minutes...")
        model = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        # Save the model locally
        model.save(str(model_path))
        st.success("Model downloaded and saved successfully!")
    else:
        model = CrossEncoder(str(model_path), device="cuda" if torch.cuda.is_available() else "cpu")
    
    return model

def re_rank_cross_encoders(documents: list[str], prompt: str) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = get_local_cross_encoder()
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])
    return relevant_text, relevant_text_ids

def process_document(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()

        # Clean up the temporary file
        os.unlink(tmp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750, chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        return splits
    return []

def add_to_vector_collection(splits: list[Document], collection_name: str):
    collection = get_vector_collection()
    collection.add(
        documents=[s.page_content for s in splits],
        ids=[f"doc_{collection_name}_{i}" for i in range(len(splits))],
    )
    st.success(f"Added {len(splits)} chunks to vector DB!")

st.set_page_config(page_title="MoHUA - PMAY Chatbot", page_icon="💬", layout="wide")

# Add custom CSS for the chat interface
st.markdown("""
<style>
    .main {
        padding: 2rem 1rem 0rem 1rem; /* Added top padding and horizontal padding */
    }
    .stChatInput {
        position: fixed;
        bottom: 0;
        background-color: #2C2C2C; /* Dark background for the input box */
        padding: 1rem;
        z-index: 100;
        width: calc(100% - 2rem); /* Account for padding */
        max-width: 700px; /* Set a max-width for centering */
        left: 50%;
        transform: translateX(-50%);
        box-sizing: border-box;
        color: white; /* White text for user input */
    }
    .stChatInput input::placeholder {
        color: #1A1C24; /* Lighter grey for placeholder text */
    }
    .chat-messages {
        padding-top: 0rem; /* Remove any top padding */
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        gap: 0.75rem;
    }
    .stChatMessageContent {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stChatMessageContent"] {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stChatMessageContent"] p {
        margin: 0;
    }
    /* Reduce space below the title */
    h1 {
        margin-bottom: 0rem; /* Set margin to 0 */
    }
    /* Target the container wrapping st.title to remove its bottom space more aggressively */
    div[data-testid="stMarkdownContainer"] {
        margin-bottom: 0rem !important;
        padding-bottom: 0rem !important;
    }
    /* Ensure the chat messages container has no top margin */
    .chat-messages {
        margin-top: 0rem !important; /* Remove any top margin */
        padding-top: 0rem; /* Keep padding-top at 0 */
    }
</style>
""", unsafe_allow_html=True)

# Create columns for centering the chat interface
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("🇮🇳 PMAY Chatbot")
    
    # Initialize session state for messages if not exists
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Create a container for the chat messages with custom class
    messages_container = st.container()
    messages_container.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    # Display previous conversation in the container
    with messages_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
        
        # JavaScript to scroll to the bottom of the chat-messages div
        st.markdown("""
            <script>
                var element = document.querySelector('.chat-messages');
                if (element) {
                    element.scrollTop = element.scrollHeight;
                }
            </script>
            """, unsafe_allow_html=True)

    messages_container.markdown('</div>', unsafe_allow_html=True)

    # Define predefined responses for greetings and introduction
    greeting_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! What can I help you with?",
        "hey": "Hey! How can I help you today?",
        "introduce yourself": "I am the PMAY MoHUA chatbot, here to assist you with information related to the Pradhan Mantri Awas Yojana and urban affairs.",
        "who are you": "I am the PMAY Chatbot, created by the Ministry of Housing and Urban Affairs (MoHUA) to assist users with queries related to the Pradhan Mantri Awas Yojana (PMAY) scheme. My goal is to provide accurate and helpful information based on the context provided.",
        "what are you": "I am the PMAY Chatbot, designed to help you with queries regarding housing and urban development, specifically related to the PMAY scheme. I can assist you with information about the application process, eligibility, and more."
    }

    # Chat input at the bottom with custom styling
    user_input = st.chat_input("Ask a question...", key="chat_input")
    if user_input:
        # Append user message first
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        # Display the user's message immediately
        with messages_container:
            st.chat_message("user").write(user_input)

        # Check for predefined greetings or introduction
        user_input_lower = user_input.lower()  # Normalize input for comparison
        if user_input_lower in greeting_responses:
            response = greeting_responses[user_input_lower]
            st.session_state["messages"].append({"role": "assistant", "content": response})
            with messages_container:
                st.chat_message("assistant").write(response)
        else:
            # Query vector DB
            results = query_collection(user_input)
            documents = results.get("documents", [])
            
            if documents and documents[0]:
                context = documents[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context, user_input)
                
                # Stream LLM response
                full_response = ""
                response_stream = call_llm(context=relevant_text, prompt=user_input)
                with messages_container:
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        for chunk in response_stream:
                            for char in chunk:
                                full_response += char
                                placeholder.markdown(full_response)
                                time.sleep(0.01)  # Adjust speed as desired
                        # Add source documents in a collapsible expander
                        with st.expander("View Source Documents", expanded=False):
                            for i, doc_id in enumerate(relevant_text_ids):
                                st.write(f"**Document {i+1}:** {doc_id}")
                                st.write(f"Content: {context[i]}")
                                if i < len(relevant_text_ids) - 1:  # Don't add separator after last document
                                    st.write("---")
                
                # Append assistant response to session state
                st.session_state["messages"].append({"role": "assistant", "content": full_response})
            else:
                warning = "No relevant documents found for your query. Please try a different question."
                st.session_state["messages"].append({"role": "assistant", "content": warning})
                with messages_container:
                    st.chat_message("assistant").write(warning)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')
