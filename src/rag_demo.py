# import streamlit as st
# import PyPDF2
# import pandas as pd
# import os
# import base64
# from io import BytesIO
# import itertools

# from dotenv import load_dotenv
# load_dotenv()  # This loads the variables from .env into the environment

# import weaviate
# import os

# client = weaviate.connect_to_wcs(
#     cluster_url=os.getenv("WCS_DEMO_URL"),  # Replace with your WCS URL
#     auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_DEMO_RO_KEY"))  # Replace with your WCS key
# )

# # Initialize session state
# if 'pdf_text' not in st.session_state:
#     st.session_state.pdf_text = ""


# def save_uploadedfile(uploadedfile):
#     with open(os.path.join("tempDir", uploadedfile.name),"wb") as f:
#         f.write(uploadedfile.getbuffer())
#     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
#     return "\n".join(text)

# def get_table_download_link(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="extracted_text.csv">Download csv file</a>'
#     return href

# # Define schema (run this once to setup your Weaviate schema)
# def setup_weaviate_schema():
#     schema = {
#         "classes": [{
#             "class": "DocumentChunk",
#             "properties": [{
#                 "name": "text",
#                 "dataType": ["text"]
#             }, {
#                 "name": "embedding",
#                 "dataType": ["vector"]
#             }]
#         }]
#     }
#     client.schema.delete_all()
#     client.schema.create(schema)

# # Optional: Uncomment the following line if you need to setup the schema
# # setup_weaviate_schema()

# def chunk_text(text, size=200):
#     words = text.split()
#     chunks = [' '.join(words[i:i + size]) for i in range(0, len(words), size)]
#     return chunks

# def embed_text(text):
#     # Use a text embedding model here. Placeholder function:
#     return client.modules.text2vec.transformers.get_vector(text)

# def index_chunks(chunks):
#     for chunk in chunks:
#         embedding = embed_text(chunk)
#         client.data_object.create({
#             "text": chunk,
#             "embedding": embedding
#         }, "DocumentChunk")

# def search_weaviate(query, top_k=5):
#     vector = embed_text(query)
#     result = client.query.get("DocumentChunk", ["text"]).with_vector(vector).with_limit(top_k).do()
#     return [hit['text'] for hit in result['data']['Get']['DocumentChunk']]

# st.title('RAG Project Demo with Weaviate')

# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# if uploaded_file is not None:
#     text = extract_text_from_pdf(uploaded_file)
#     chunks = chunk_text(text)
#     index_chunks(chunks)
#     st.session_state.pdf_text = text
#     st.text_area("Extracted Text", text, height=300)

# query = st.text_input("Enter your query here")
# if st.button('Retrieve Information'):
#     if query and st.session_state.pdf_text:
#         results = search_weaviate(query)
#         st.write("Search Results:")
#         for result in results:
#             st.text(result)
#     else:
#         st.warning("Please upload a PDF and enter a query.")






# #TODO
# # import streamlit as st
# # import os
# # import dotenv
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_community.vectorstores import Weaviate
# # from langchain_openai import OpenAIEmbeddings
# # import database

# # # Load environment variables
# # dotenv.load_dotenv()
# # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# # # Initialize Weaviate client
# # client = database.create_client()

# # # Streamlit UI
# # st.title('Document Processing and Search with Weaviate')

# # uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
# # if uploaded_file is not None:
# #     loader = PyPDFLoader(uploaded_file, extract_images=True)
# #     documents = loader.load()

# #     chunked_docs = []
# #     for doc in documents:
# #         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250, length_function=len, is_separator_regex=False)
# #         chunks = splitter.create_documents([doc.page_content])
# #         chunked_docs.extend(chunks)

# #     embeddings = OpenAIEmbeddings()

# #     try:
# #         stored_data = Weaviate.from_documents(
# #             chunked_docs,
# #             embeddings,
# #             client=client,
# #             by_text=False,
# #             index_name="Article",
# #             text_key="content",
# #         )
# #         st.success("Data indexed successfully!")
# #     except Exception as e:
# #         st.error(f"Indexing failed: {e}")

# # query = st.text_input("Enter your search query")
# # if st.button('Search'):
# #     vector = embeddings.get_vector(query)
# #     results = client.query.get("Article", ["content"]).with_vector(vector).with_limit(5).do()
# #     if results:
# #         for result in results['data']['Get']['Article']:
# #             st.text(result['content'])
# #     else:
# #         st.warning("No results found.")



# import streamlit as st
# import PyPDF2
# import pandas as pd
# import os
# import base64
# import openai
# from dotenv import load_dotenv
# import weaviate

# # Load environment variables
# load_dotenv()

# # Initialize Weaviate client
# client = weaviate.connect_to_wcs(
#     cluster_url=os.getenv("WCS_DEMO_URL"),
#     auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_DEMO_RO_KEY"))
# )

# # OpenAI API Key
# openai.api_key = os.getenv("OPENAI_API_KEY")



# def embed_text(text):
#     try:
#         response = openai.Embedding.create(
#             input=text,
#             engine="text-similarity-davinci-001"
#         )
#         embedding = response['data'][0]['embedding']
#         return embedding
#     except Exception as e:
#         print("Failed to get embeddings:", e)
#         return None

# # Rest of your functions remain the same...

# def index_chunks(chunks):
#     for chunk in chunks:
#         embedding = embed_text(chunk)
#         if embedding is not None:
#             client.data_object.create({
#                 "text": chunk,
#                 "embedding": embedding
#             }, "DocumentChunk")
#         else:
#             st.error("Failed to embed and index chunk")

# # Streamlit UI code (no changes needed here, remains as is)
# st.title('RAG Project Demo with Weaviate')
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# if uploaded_file is not None:
#     text = extract_text_from_pdf(uploaded_file)
#     chunks = chunk_text(text)
#     index_chunks(chunks)
#     st.session_state.pdf_text = text
#     st.text_area("Extracted Text", text, height=300)

# query = st.text_input("Enter your query here")
# if st.button('Retrieve Information'):
#     if query and st.session_state.pdf_text:
#         results = search_weaviate(query)
#         st.write("Search Results:")
#         for result in results:
#             st.text(result)
#     else:
#         st.warning("Please upload a PDF and enter a query.")






import streamlit as st
import PyPDF2
import pandas as pd
import os
import base64
import openai
from dotenv import load_dotenv
import weaviate

# Load environment variables
load_dotenv()

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


def chunk_text(text, size=200):
    words = text.split()
    chunks = [' '.join(words[i:i + size]) for i in range(0, len(words), size)]
    return chunks

def search_weaviate(query, top_k=5):
    vector = embed_text(query)
    result = client.query.get("DocumentChunk", ["text"]).with_vector(vector).with_limit(top_k).do()
    return [hit['text'] for hit in result['data']['Get']['DocumentChunk']]

# Initialize Weaviate client
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WCS_DEMO_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_DEMO_RO_KEY"))
)

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")


def embed_text(text):
    try:
        response = openai.Embedding.create(
            input=text,
            engine="text-similarity-davinci-001"
        )
        if response and 'data' in response and response['data']:
            embedding = response['data'][0]['embedding']
            return embedding
        else:
            st.error(f"No embedding data returned: {response}")
            return None
    except Exception as e:
        st.error(f"Failed to get embeddings: {e}")
        return None

def index_chunks(chunks):
    for chunk in chunks:
        embedding = embed_text(chunk)
        if embedding is not None:
            try:
                client.data_object.create({
                    "text": chunk,
                    "embedding": embedding
                }, "DocumentChunk")
                st.success("Chunk indexed successfully")
            except Exception as e:
                st.error(f"Failed to create data object in Weaviate: {e}")
        else:
            st.error("Failed to embed and index chunk")

# Streamlit UI setup remains unchanged
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




# import streamlit as st
# import PyPDF2
# import os
# import base64
# from dotenv import load_dotenv
# import weaviate
# # from transformers import AutoTokenizer, AutoModel
# # import torch

# # Load environment variables
# load_dotenv()

# # Initialize Weaviate client
# client = weaviate.connect_to_wcs(
#     cluster_url=os.getenv("WCS_DEMO_URL"),
#     auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_DEMO_RO_KEY"))
# )

# # Initialize HuggingFace model and tokenizer
# # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# # model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import numpy as np

# # Initialize your embedding model
# model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")


# def embed_text(text):
#     """Generate embeddings using HuggingFace transformer model."""
#     # with torch.no_grad():
#     #     encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
#     #     model_output = model(**encoded_input)
#     #     embeddings = model_output.pooler_output
#     #     return embeddings[0].numpy()
#     return model.get_text_embedding(text)

# def extract_text_from_pdf(pdf_file):
#     """Extract text from uploaded PDF file."""
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
#     return "\n".join(text)

# def chunk_text(text, size=200):
#     """Chunk extracted text into smaller segments."""
#     words = text.split()
#     chunks = [' '.join(words[i:i + size]) for i in range(0, len(words), size)]
#     return chunks

# def index_chunks(chunks):
#     """Index chunks into Weaviate with embeddings."""
#     for chunk in chunks:
#         embedding = embed_text(chunk)
#         client.data_object.create({
#             "text": chunk,
#             "embedding": embedding.tolist()
#         }, "DocumentChunk")

# def search_weaviate(query, top_k=5):
#     """Search indexed data in Weaviate based on query."""
#     vector = embed_text(query)
#     result = client.query.get("DocumentChunk", ["text"]).with_vector(vector.tolist()).with_limit(top_k).do()
#     return [hit['text'] for hit in result['data']['Get']['DocumentChunk']]

# # Streamlit UI setup
# st.title('Document Analysis and Search')
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# if uploaded_file is not None:
#     text = extract_text_from_pdf(uploaded_file)
#     chunks = chunk_text(text)
#     index_chunks(chunks)
#     st.session_state.pdf_text = text
#     st.text_area("Extracted Text", text, height=300)

# query = st.text_input("Enter your query here")
# if st.button('Retrieve Information'):
#     if query and st.session_state.pdf_text:
#         results = search_weaviate(query)
#         st.write("Search Results:")
#         for result in results:
#             st.text(result)
#     else:
#         st.warning("Please upload a PDF and enter a query.")
