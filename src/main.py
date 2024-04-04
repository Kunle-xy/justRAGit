import os
import database
import dotenv
import pandas as pd
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings

client = database.create_client()

def clean():
    return client.schema.delete_class("Article")


if client.is_ready():
    print("Weaviate is ready!")
else:
    print("Weaviate is not ready!")


dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



def docLoader(filename):
    loader = PyPDFLoader(filename, extract_images=True)
    documents = loader.load()
    return documents

def chunkRecursively(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    res = text_splitter.create_documents([text])
    return res


def index(filename, chunk_size, chunk_overlap):
    chunked_doc = []
    for _, doc in enumerate(docLoader(filename)):
        chunked_doc.append(chunkRecursively(doc.page_content, chunk_size, chunk_overlap)) # 25% overlap

    vector_store = []
    for item in chunked_doc:
        for doc in item:
            vector_store.append(doc)


    embeddings = OpenAIEmbeddings()
    try:
            storedData = Weaviate.from_documents(
                vector_store,
                embeddings,
                client=client,
                by_text=False,
                index_name="Article",
                text_key="content",
                )
            
            if storedData:
                print("Data indexed successfully!")

    except Exception as e:
        print(e)

    finally:
        print("All data indexed successfully!")



def args():
    parser = argparse.ArgumentParser(description="Upload and Index file to Weaviate")
    parser.add_argument("--pdf_file", help="Input file", required=False, type=str)
    parser.add_argument("--chunk_size", help="Chunk size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", help="Chunk overlap", type=int, default=250)
    return parser.parse_args()


if  __name__ == "__main__":
    args = args()
    # print(args)

    if args.pdf_file:
        index(args.pdf_file, args.chunk_size, args.chunk_overlap)
    else:
        print("No input file provided")
        exit(1)
