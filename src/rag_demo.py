import streamlit as st
import PyPDF2
import pandas as pd
import os
import base64
from io import BytesIO
import itertools

from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env into the environment

import weaviate
import os

client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WCS_DEMO_URL"),  # Replace with your WCS URL
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_DEMO_RO_KEY"))  # Replace with your WCS key
)

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""


def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir", uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
    return "\n".join(text)

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="extracted_text.csv">Download csv file</a>'
    return href

# Define schema (run this once to setup your Weaviate schema)
def setup_weaviate_schema():
    schema = {
        "classes": [{
            "class": "DocumentChunk",
            "properties": [{
                "name": "text",
                "dataType": ["text"]
            }, {
                "name": "embedding",
                "dataType": ["vector"]
            }]
        }]
    }
    client.schema.delete_all()
    client.schema.create(schema)

# Optional: Uncomment the following line if you need to setup the schema
# setup_weaviate_schema()

def chunk_text(text, size=200):
    words = text.split()
    chunks = [' '.join(words[i:i + size]) for i in range(0, len(words), size)]
    return chunks

def embed_text(text):
    # Use a text embedding model here. Placeholder function:
    return client.modules.text2vec.transformers.get_vector(text)

def index_chunks(chunks):
    for chunk in chunks:
        embedding = embed_text(chunk)
        client.data_object.create({
            "text": chunk,
            "embedding": embedding
        }, "DocumentChunk")

def search_weaviate(query, top_k=5):
    vector = embed_text(query)
    result = client.query.get("DocumentChunk", ["text"]).with_vector(vector).with_limit(top_k).do()
    return [hit['text'] for hit in result['data']['Get']['DocumentChunk']]

st.title('RAG Project Demo with Weaviate')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    index_chunks(chunks)
    st.session_state.pdf_text = text
    st.text_area("Extracted Text", text, height=300)

query = st.text_input("Enter your query here")
if st.button('Retrieve Information'):
    if query and st.session_state.pdf_text:
        results = search_weaviate(query)
        st.write("Search Results:")
        for result in results:
            st.text(result)
    else:
        st.warning("Please upload a PDF and enter a query.")


