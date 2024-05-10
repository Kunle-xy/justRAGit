import streamlit as st
import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import dotenv

import database
client = database.create_client()

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
collection_name = "Article"


def embed_text(text):
    api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "text-embedding-3-small",
        "input": text,
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        embeddings = response.json()['data']
        return embeddings
    else:
        st.error(f"Failed to get embeddings: {response.status_code}, {response.text}")
        return []


def docLoader(filename):
    loader = PyPDFLoader(filename, extract_images=True)
    documents = loader.load()
    return documents


def chunkRecursively(text, chunk_size=1000, chunk_overlap=250):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    res = text_splitter.create_documents([text])
    return res


def index_pdf(filename):
    chunked_doc = []
    for _, doc in enumerate(docLoader(filename)):
        chunked_doc.append(chunkRecursively(doc.page_content))

    vector_store = []
    for item in chunked_doc:
        for doc in item:
            vector_store.append(doc.page_content)

    chunk_class = {
        "class": collection_name,
        "properties": [
            {
                "name": "chunk",
                "dataType": ["text"],
            },
            {
                "name": "chunk_index",
                "dataType": ["int"],
            }
        ],
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "model": "text-embedding-3-small",
                "dimensions": 1536,
                "tokenization": True,
            }
        }
    }

    if client.schema.exists(collection_name):
        client.schema.delete_class(collection_name)
    client.schema.create_class(chunk_class)

    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for i, chunk in enumerate(vector_store):
            data_object = {
                "chunk": chunk,
                "chunk_index": i
            }
            batch.add_data_object(data_object=data_object, class_name=collection_name)

    st.success("Data indexed successfully!")
    response = client.query.aggregate(collection_name).with_meta_count().do()
    #st.write(response)


def queries(query, top_k=5, top_p=4):
    q_embeddings = {'vector': embed_text(query)[0]['embedding']}
    result = client.query.get(collection_name, ["chunk"]).with_near_vector(q_embeddings).with_limit(top_k).with_additional(['certainty']).do()
    results_with_scores = []

    try:
        for res in result['data']['Get'][collection_name]:
            chunk = res['chunk']
            certainty = res.get('additional', {}).get('certainty', 0)
            results_with_scores.append((chunk, certainty))
    except KeyError as e:
        st.error(f"Key error encountered: {e}")
        return "Failed to process results due to key error."

    sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    top_p_results = sorted_results[:top_p]
    context = "\n\n".join([res[0] for res in top_p_results])

    return context


def answer_query(query, context):
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    rag_chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(f"{query}")


st.title("PDF Index and Query System Using RAG")

st.subheader("Upload and Index PDF File")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Fixed chunk size and chunk overlap values
fixed_chunk_size = 1000
fixed_chunk_overlap = 250

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    index_pdf(uploaded_file.name)

st.subheader("Query System")
query = st.text_input("Enter your query", value="")

# Add a button to submit the query
if st.button("Submit Query"):
    if query:
        # Fixed values for top_k and top_p
        top_k = 5
        top_p = 4

        context = queries(query, top_k, top_p)


        answer = answer_query(query, context)
        st.write("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a query before submitting.")
