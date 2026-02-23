#!/usr/bin/env python3
"""Simple test script: load sample_document.txt, find relevant chunks, and
attempt to generate an answer using the OpenAI v1 client if available.
"""
import os
import re
from dotenv import load_dotenv

# load env
load_dotenv()

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'sample_document.txt')

def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def find_relevant_chunks(question, chunks, top_k=3):
    question_words = set(re.findall(r"\\b\\w+\\b", question.lower()))
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r"\\b\\w+\\b", chunk.lower()))
        intersection = len(question_words.intersection(chunk_words))
        union = len(question_words.union(chunk_words))
        score = intersection / union if union > 0 else 0
        scored_chunks.append((score, chunk, i))
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return scored_chunks[:top_k]


def main():
    if not os.path.exists(SAMPLE_PATH):
        print('sample_document.txt not found')
        return

    with open(SAMPLE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = chunk_text(text)
    question = 'What are the applications of AI in healthcare?'

    relevant = find_relevant_chunks(question, chunks)
    print(f'Relevant chunks (top {len(relevant)}):')
    for score, chunk, idx in relevant:
        print('---')
        print(f'Relevance: {score:.3f}, chunk index: {idx}')
        print(chunk[:400].replace('\n',' '))

    # Try OpenAI client if available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('\nOPENAI_API_KEY not set. Skipping API call.')
        return

    # Use REST fallback to call OpenAI Chat Completions endpoint directly
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('\nOPENAI_API_KEY not set. Skipping API call.')
        return

    try:
        import requests, json
    except Exception as e:
        print(f"requests not available: {e}\nSkipping API call.")
        return

    context = '\n\n'.join([c for _, c, _ in relevant])
    prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say so.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""

    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 300,
        'temperature': 0
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if r.status_code != 200:
            print(f'API error {r.status_code}: {r.text}')
            return
        j = r.json()
        try:
            content = j['choices'][0]['message']['content']
        except Exception:
            content = j['choices'][0].get('text')

        print('\nModel answer:')
        print(content or 'No content returned')
    except Exception as e:
        print(f'Error calling OpenAI REST API: {e}')

if __name__ == '__main__':
    main()
