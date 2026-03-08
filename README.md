# Document Q&A AI Agent

A powerful AI-powered question-answering system that allows users to upload documents and ask context-aware questions about their content.

## Features

- 📁 **Document Upload**: Support for PDF, TXT, and DOCX files
- 🤖 **AI-Powered Q&A**: Uses OpenAI's GPT models for intelligent answers
- 🔍 **Context-Aware**: Retrieves relevant information from uploaded documents using keyword-based similarity
- 🎨 **Modern UI**: Clean and intuitive Streamlit interface
- 📊 **Source Tracking**: Shows which parts of documents were used for answers

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd qa_agent
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key:**
   - Open the `.env` file
   - Replace `your_openai_api_key_here` with your actual OpenAI API key
   - You can get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```
   Or use the provided script:
   ```bash
   ./run.sh
   ```

2. **Access the application:**
   - Open your browser and go to `http://localhost:8501`

3. **Upload documents:**
   - Use the sidebar to upload PDF, TXT, or DOCX files
   - Click "Process Documents" to analyze and index the content

4. **Ask questions:**
   - Enter your question in the main text field
   - Click "Get Answer" to receive context-aware responses
   - View source information that shows document relevance

## How It Works

1. **Document Processing**: Uploaded documents are parsed and text is extracted
2. **Text Chunking**: Documents are split into manageable chunks with overlap
3. **Question Analysis**: User questions are analyzed for keywords
4. **Similarity Matching**: Chunks are ranked by keyword similarity to the question
5. **Answer Generation**: OpenAI's GPT model generates answers using the most relevant chunks

## Supported File Types

- **PDF**: Portable Document Format files
- **TXT**: Plain text files
- **DOCX**: Microsoft Word documents

## Architecture

- **Frontend**: Streamlit web application
- **AI Models**: OpenAI GPT-3.5-turbo for text generation
- **Document Processing**: PyPDF2 for PDFs, docx2txt for Word documents
- **Text Analysis**: Keyword-based similarity matching
- **Environment**: Python virtual environment for dependency isolation

## Customization

You can customize the application by:

- Modifying chunk size and overlap in `app.py`
- Changing the number of retrieved chunks (top_k parameter)
- Using different OpenAI models
- Adding support for more document types

## API Key Setup

Make sure to set your OpenAI API key in the `.env` file:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Troubleshooting

- **API Key Issues**: Make sure your OpenAI API key is correctly set in the `.env` file
- **Document Loading Errors**: Check that your documents are not corrupted and are in supported formats
- **Memory Issues**: For large documents, consider reducing chunk size
- **Import Errors**: Make sure you're running the application from the activated virtual environment

## Project Structure

```
qa_agent/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (API keys)
├── run.sh             # Startup script
├── test_imports.py    # Import verification script
└── README.md          # This file
```

## License

This project is open source and available under the MIT License.