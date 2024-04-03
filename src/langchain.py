import os
import json
import database
import dotenv
# Standard library imports


# Third-party imports
import weaviate
import weaviate.classes as wvc
import requests
import pytesseract
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

# Local application/library specific imports
import langchain
langchain.debug = False
langchain.verbose = False
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PyMuPDFLoader
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage



# Set the tesseract command path
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"





client = database.create_client()
dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# print(os.getenv("OPENAI_API_KEY"))

# loader = PyMuPDFLoader("../sample.pdf", extract_images = True)   
# documents = loader.load()

# print(documents[2], len(documents))

filename = "../sample.pdf"
# Extracts the elements from the PDF
# elements = partition_pdf(
#     filename=filename,
#     # url=None,
#     # Unstructured Helpers

#     infer_table_structure=True,
#     extract_images_in_pdf=True,

#     chunking_strategy="by_title",
#     max_characters=1000,
#     new_after_n_chars=500,
#     overlap = 100,
#     overlap_all = True,

#     image_output_dir_path = "./images",
# )

# appending texts and tables from the pdf file
# def data_category(raw_pdf_elements): # we may use decorator here
#     tables = []
#     texts = []
#     for element in raw_pdf_elements:
#         if "unstructured.documents.elements.Table" in str(type(element)):
#            tables.append(str(element))
#         elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
#            texts.append(str(element))
#     data_categories = [texts,tables]
#     return data_categories

# data_categories = data_category(elements)
# texts = data_categories[0]
# tables = data_categories[1]
# print(len(texts), len(tables), len(elements))


# function to take tables as input and then summarize them
def tables_summarize(tables):
    prompt_text = """You are an assistant tasked with summarizing tables. \
                    Give a concise summary of the table. Table chunk: {element} """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    
    return table_summaries


result = tables_summarize(["Table 1: This is a table", "Table 2: This is another table"])

print(result)
# table_summaries = tables_summarize(tables)
# text_summaries = texts

# print(table_summaries)

# def encode_image(image_path):
#     ''' Getting the base64 string '''
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# def image_captioning(img_base64, prompt):
#     ''' Image summary '''
#     chat = ChatOpenAI(model="gpt-4-vision-preview",
#                       max_tokens=1024)

#     msg = chat.invoke(
#         [
#             HumanMessage(
#                 content=[
#                     {"type": "text", "text":prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{img_base64}"
#                         },
#                     },
#                 ]
#             )
#         ]
#     )
#     return msg.content


# # Store base64 encoded images
# img_base64_list = []

# # Store image summaries
# image_summaries = []

# # Prompt : Our prompt here is customized to the type of images we have which is chart in our case
# prompt = "Describe the image in detail. Be specific about graphs, such as bar plots."
# path = "./images"

# # Read images, encode to base64 strings
# for img_file in sorted(os.listdir(path)):
#     if img_file.endswith('.jpg'):
#         img_path = os.path.join(path, img_file)
#         base64_image = encode_image(img_path)
#         img_base64_list.append(base64_image)
#         img_capt = image_captioning(base64_image,prompt)
#         time.sleep(60)
#         image_summaries.append(image_captioning(img_capt,prompt))


# def split_image_text_types(docs):
#     ''' Split base64-encoded images and texts '''
#     b64 = []
#     text = []
#     for doc in docs:
#         try:
#             b64decode(doc)
#             b64.append(doc)
#         except Exception as e:
#             text.append(doc)
#     return {
#         "images": b64,
#         "texts": text
#     }



































































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