import os
import asyncio
import nest_asyncio
import requests
import streamlit as st
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import fitz  # pymupdf for PDF processing

# Set up the working directory
WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

nest_asyncio.apply()

# Initialize Hugging Face model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    url = "https://api.hyperbolic.xyz/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJvbWJha2FsZTIyQGdtYWlsLmNvbSIsImlhdCI6MTczMjY0MDMxMH0.CCgwKM-3WagOVXXlcBMsf-IzUWrOmFz3ZTz9SLVT0fo"  # Replace with your actual token
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "max_tokens": 131072,
        "temperature": 0.1,
        "top_p": 1
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Check for successful response
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response text: {response.text}")
            return ""

        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    except requests.exceptions.RequestException as e:
        print(f"Request exception occurred: {e}")
        return ""
    except ValueError as e:
        print(f"JSON decoding error occurred: {e}")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Average pool the token embeddings

    return embeddings.numpy()

async def get_embedding_dim():
    test_text = text
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pymupdf (fitz)."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit App Interface
def run_streamlit_app():
    st.title("PDF Query System with RAG and Hugging Face")
    
    # File upload widget for the PDF
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if pdf_file is not None:
        pdf_text = extract_text_from_pdf(pdf_file)
        st.write("PDF text extracted successfully!")
        
        # Initialize the RAG model for querying
        embedding_dimension = 768  # Example fixed value, modify as needed
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,
                func=embedding_func,
            ),
        )

        # Insert the extracted PDF text into RAG
        asyncio.run(rag.ainsert(pdf_text))
        
        # Query Input
        prompt = st.text_input("Enter your query:", "")
        
        if st.button("Query"):
            if prompt:
                try:
                    # Perform a query
                    response = asyncio.run(rag.aquery(prompt, param=QueryParam(mode="naive")))
                    st.write("Response from the system:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error occurred: {e}")
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    run_streamlit_app()
