from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_text_splitters import CharacterTextSplitter
from pdfminer.high_level import extract_text

from langchain_community.embeddings import HuggingFaceEmbeddings
from src import database
import argparse
import os


def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def clean(class_name="TextChunk"):
    client = database.create_client()
    query = f"{{Get{{ {class_name} {{uuid}}}}}}"
    response = client.query.raw(query)
       # Extract UUIDs from the query response
    uuids = [obj['uuid'] for obj in response['data']['Get'][class_name]]

    # Delete objects one by one
    for uuid in uuids:
        client.data_object.delete(uuid, class_name)
        print(f"Deleted {class_name} object with UUID: {uuid}")


def chunk_text(text, chunk_size=1000):
    """
    Splits the text into chunks of a specific size.
    """
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def upload_pdf_to_weaviate(pdf_path, chunk_size=1000):
    """
    Extracts text from a PDF, chunks it, and uploads each chunk to Weaviate.
    """
    # Extract text from the PDF
    extracted_text = extract_text(pdf_path)

    # Initialize Weaviate client
    client = database.create_client()

    # Define schema - this should ideally be done outside this function and only once.
    schema = {
        "classes": [{
            "class": "TextChunk",
            "description": "A chunk of text from a document",
            "properties": [{
                "name": "content",
                "dataType": ["text"],
                "description": "The content of the text chunk",
            }]
        }]
    }

    try:
        client.schema.create(schema)
    except Exception as e:
        print("Schema might already exist or another error occurred:", e)
    
    # Chunk the text and upload each chunk
    for chunk in chunk_text(extracted_text, chunk_size):
        text_chunk = {
            "content": chunk
        }
        client.data_object.create(text_chunk, "TextChunk")

    print("All chunks uploaded to Weaviate.")

def upload(file_path:str, chunk_size: int=1000, chunk_overlap:int=0):
    client = database.create_client()
    file_name = os.path.basename(file_path)

    # check if file exist
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    

    string_data = extract_text_from_pdf(file_path)
    documents = [{'content': string_data}]

    # tmp_file = os.path.splitext(file_name)[0] + ".txt"
    # with open(tmp_file, "w") as file:
    #     file.write(string_data)
    
    # loader = TextLoader(tmp_file)
    # documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    for chunk in chunks:
        embedding = embeddings.encode(chunk["content"])
        document = {
            "content": chunk["content"],
            "embedding": embedding.tolist()
        }

        client.data_object.create("Document", document)

    counts = client.query.aggregate("Document").with_meta_count().do()
    counts = counts['data']['Aggregate']['Document'][0]['meta']['count']

    print(f"Document {file_name} uploaded to Weaviate. In total: {counts} documents in Weaviate.")


def upload_pdf(file_path:str, chunk_size: int=1000, chunk_overlap:int=0):
    client = database.create_client()
    client.schema.delete_all()
    schema = {
    "classes": [
        {
            "class": "Chunked Document",
            "description": "A class to store text documents",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "A chunk of the document",
                }
            ]
        }
        ]
    }
    client.schema.create(schema)
    extracted_text = extract_text_from_pdf(file_path)
    document = {
    "content": extracted_text
                }
    client.data_object.create("Document", document)

    return f"Document {file_path} uploaded to Weaviate."



def args():
    parser = argparse.ArgumentParser(description="Upload and Index file to Weaviate")
    parser.add_argument("--clean", help="Clean Weaviate", action="store_true")
    parser.add_argument("--pdf_file", help="Input file", required=False, type=str)
    parser.add_argument("--chunk_size", help="Chunk size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", help="Chunk overlap", type=int, default=0)
    return parser.parse_args()


if  __name__ == "__main__":
    args = args()
    print(args)
    if args.clean:
        clean()
    elif args.pdf_file:
        upload_pdf_to_weaviate(args.pdf_file, args.chunk_size)
    else:
        print("No input file provided")
        exit(1)
