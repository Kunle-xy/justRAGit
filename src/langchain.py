import weaviate
import weaviate.classes as wvc
import os
import requests
import json
import database
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"




client = database.create_client()

# loader = PyMuPDFLoader("../sample.pdf", extract_images = True)   
# documents = loader.load()

# print(documents[2], len(documents))

filename = "../sample.pdf"
# Extracts the elements from the PDF
elements = partition_pdf(
    filename=filename,
    # url=None,
    # Unstructured Helpers
    strategy="hi_res", 
    infer_table_structure=True, 
    model_name="yolox",
    max_characters=1500,
    new_after_n_chars=1000,
    overlap = 100,
    overlap_all = True,
)

print(elements[17], elements[43], elements[55])

#Add data to Weaviate
# try:
#         questions = client.collections.create(
#             name="Questionzs",
#             vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
#             generative_config=wvc.config.Configure.Generative.openai()  # Ensure the `generative-openai` module is used for generative queries
        
#     )

#     resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
#     data = json.loads(resp.text)  # Load data

#     question_objs = list()
#     for i, d in enumerate(data):
#         question_objs.append({
#             "answer": d["Answer"],
#             "question": d["Question"],
#             "category": d["Category"],
#         })

#     questions = client.collections.get("Questionzs")
#     questions.data.insert_many(question_objs) 
#     print(questions)

# finally:
#     client.close()


# Semantic search


# try:
#   # Replace with your code. Close client gracefully in the finally block.
#     questions = client.collections.get("Question")

#     response = questions.query.near_text(
#         query="biology",
#         limit=2
#     )

#     print(response.objects[0].properties, response.objects)  # Inspect the first object

# finally:
#     client.close()


# Generative
# questions = client.collections.get("Question")

# response = questions.generate.near_text(
#     query="biology",
#     limit=1,  # limit the number of results
#     # single_prompt="Explain {answer} as you might to a five-year-old."
#     grouped_task="Write a tweet with emojis about these facts."

# )

# print(response.objects[1].generated) 
# for idx, obj in enumerate(response.objects):
#     print(f"{idx}: {obj.generated}")  # Inspect the first object


#DELETE SCHEMA
# try:
#     schema = client.collections.list_all(simple=True)  # Use `simple=False` to get comprehensive information
#     for s in schema:

#         print(s)

# finally:
#     client.close()