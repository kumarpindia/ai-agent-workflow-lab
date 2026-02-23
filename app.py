import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
import docx2txt
import tempfile
import re
import json

# requests fallback for OpenAI REST API when client library causes issues
try:
    import requests
except Exception:
    requests = None

# Load environment variables
load_dotenv()
# Create OpenAI client (require API key)
api_key = os.getenv("OPENAI_API_KEY")
api_key_missing = False
client_init_error = None
if not api_key:
    api_key_missing = True
    client = None
else:
    # Try to create OpenAI client; if the client library causes init errors,
    # fall back to None so the app uses the REST fallback path.
    try:
        client = OpenAI()
    except Exception as e:
        client = None
        client_init_error = str(e)

# Set page config
st.set_page_config(page_title="Document Q&A Agent", page_icon="📚", layout="wide")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        text = docx2txt.process(file)
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_txt(file):
    """Extract text from TXT file"""
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT: {e}")
        return ""

def load_document(uploaded_file):
    """Load document based on file type"""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    elif file_extension in ['docx', 'doc']:
        return extract_text_from_docx(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def find_relevant_chunks(question, chunks, top_k=3):
    """Find most relevant chunks for the question using simple keyword matching"""
    question_words = set(re.findall(r'\b\w+\b', question.lower()))

    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
        # Calculate Jaccard similarity
        intersection = len(question_words.intersection(chunk_words))
        union = len(question_words.union(chunk_words))
        score = intersection / union if union > 0 else 0
        scored_chunks.append((score, chunk, i))

    # Sort by score and return top_k chunks
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return scored_chunks[:top_k]

def generate_answer(question, relevant_chunks):
    """Generate answer using OpenAI GPT"""
    # If mock mode is enabled in the Streamlit UI, return a simulated answer
    try:
        try:
            if st.session_state.get('mock_mode'):
                snippets = [chunk[:500] for _, chunk, _ in relevant_chunks]
                joined = "\n\n---\n\n".join(snippets) if snippets else "(no relevant snippets)"
                return f"MOCK ANSWER (UI TEST MODE):\n\nThis is a simulated response based on the retrieved snippets.\n\n{joined}"
        except Exception:
            # If Streamlit session state isn't available, continue to attempt real call
            pass
        # Prepare context from relevant chunks
        context = "\n\n".join([chunk for _, chunk, _ in relevant_chunks])

        prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

        # Ensure client is configured
        if client is None:
            return "Error generating answer: OPENAI_API_KEY is not configured."

        # Try using OpenAI client if available
        if client is not None:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0
                )

                # Try multiple extraction patterns
                try:
                    content = response.choices[0].message.content
                except Exception:
                    try:
                        content = response.choices[0]["message"]["content"]
                    except Exception:
                        try:
                            content = response.choices[0]["text"]
                        except Exception:
                            content = None

                if content:
                    return content.strip()
            except Exception:
                # fall through to REST fallback
                pass

        # Fallback: use direct REST call to OpenAI Chat Completions
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error generating answer: OPENAI_API_KEY is not configured."

        if requests is None:
            return "Error generating answer: 'requests' library is not available for REST fallback."

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0
        }

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if resp.status_code != 200:
            return f"Error generating answer: API returned status {resp.status_code}: {resp.text}"

        j = resp.json()
        # Extract text from response JSON
        try:
            content = j["choices"][0]["message"]["content"]
        except Exception:
            try:
                content = j["choices"][0]["text"]
            except Exception:
                content = None

        if not content:
            return "Error generating answer: no content returned from the model (REST)."

        return content.strip()

    except Exception as e:
        return f"Error generating answer: {str(e)}"

def process_documents(uploaded_files):
    """Process uploaded documents"""
    all_text = ""
    document_names = []

    for file in uploaded_files:
        with st.spinner(f"Loading {file.name}..."):
            text = load_document(file)
            if text:
                all_text += text + "\n\n"
                document_names.append(file.name)

    if not all_text.strip():
        st.error("No text could be extracted from the uploaded documents.")
        return False

    # Chunk the text
    with st.spinner("Processing documents..."):
        chunks = chunk_text(all_text)

    # Store in session state
    st.session_state.documents = chunks
    st.session_state.document_names = document_names
    st.session_state.documents_processed = True

    st.success(f"Successfully processed {len(document_names)} documents with {len(chunks)} text chunks!")
    return True

def main():
    st.title("📚 Document Q&A AI Agent")
    st.markdown("Upload your documents and ask questions to get context-aware answers!")

    # Show API key / client warnings (avoid calling before set_page_config)
    if api_key_missing:
        st.warning("OPENAI_API_KEY not found. Set it in the .env file to enable the AI.")
    if client_init_error:
        st.warning(f"OpenAI client initialization failed; falling back to REST. ({client_init_error})")

    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        # Mock mode toggle for UI testing without calling OpenAI
        st.checkbox("Mock mode (no API calls)", value=False, key="mock_mode", help="When enabled, the app will not call OpenAI and will generate mock answers for testing.")
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=['pdf', 'txt', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files"
        )

        if uploaded_files and st.button("Process Documents", type="primary"):
            process_documents(uploaded_files)

        # Show processing status
        if st.session_state.documents_processed:
            st.success("✅ Documents processed and ready for Q&A!")
            st.info(f"📄 {len(st.session_state.document_names)} documents loaded")
        else:
            st.info("⬆️ Upload and process documents to start asking questions")

    # Main Q&A interface
    if st.session_state.documents_processed:
        st.header("❓ Ask Questions")

        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about the documents?"
        )

        if question and st.button("Get Answer", type="primary"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # Find relevant chunks
                    relevant_chunks = find_relevant_chunks(question, st.session_state.documents)

                    if not relevant_chunks or relevant_chunks[0][0] == 0:
                        st.warning("No relevant information found in the documents for this question.")
                        return

                    # Generate answer
                    answer = generate_answer(question, relevant_chunks)

                    # Display answer
                    st.subheader("💡 Answer:")
                    st.write(answer)

                    # Display source information
                    with st.expander("📄 Source Information"):
                        st.write(f"**Documents processed:** {', '.join(st.session_state.document_names)}")
                        st.write(f"**Relevant chunks found:** {len(relevant_chunks)}")

                        for i, (score, chunk, chunk_idx) in enumerate(relevant_chunks, 1):
                            st.markdown(f"**Relevant Section {i}:** (Relevance: {score:.2f})")
                            # Show first 300 characters of the chunk
                            preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                            st.text_area(
                                f"Content preview {i}",
                                preview,
                                height=100,
                                key=f"preview_{i}",
                                disabled=True
                            )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if "api" in str(e).lower():
                        st.info("Make sure your OpenAI API key is set correctly in the .env file")

    else:
        st.info("👈 Please upload and process some documents first using the sidebar.")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and OpenAI*")

if __name__ == "__main__":
    main()