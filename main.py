from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_community.embeddings import HuggingFaceEmbeddings
from src import database
import argparse
import shutils

def clean():
    client = database.create_client()
    db = Weaviate(client)
    db.delete_all_documents()
    print("All documents deleted from Weaviate")


def upload(file_path:str, chunk_size: int=1000, chunk_overlap:int=0):
    file_name = os.path.basename(file_path)

    # check if file exist
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    # check if file is a pdf
    if not file_name.endswith(".pdf"):
        raise ValueError(f"File {file_name} is not a pdf file")
    
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    client = database.create_client()
    db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)

    # file_name = os.path.splitext(file_name)[0]
    # with open(file_path, "rb") as file:
    #     file_content = file.read()
    # client = database.create_client()
    # db = Weaviate(client)
    # db.upload_document(file_name, file_content)

    # count the number of uploaded documents
    counts = client.query.aggregate("Document").with_meta_count().do()
    counts = counts['data']['Aggregate']['Document'][0]['meta']['count']



    print(f"Document {file_name} uploaded to Weaviate. In total: {counts} documents in Weaviate.")


def args():
    parser = argparse.ArgumentParser(description="Upload and Index file to Weaviate")
    parser.add_argument("--clean", help="Clean Weaviate", action="store_true")
    parser.add_argument("--pdf_file", help="Input file", required=True)
    parser.add_argument("--chunk_size", help="Chunk size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", help="Chunk overlap", type=int, default=0)
    return parser.parse_args()


if  __name__ == "__main__":
    args = args()
    if args.clean:
        clean()
    else:
        upload(args.pdf_file, args.chunk_size, args.chunk_overlap)
