from langchain_community.document_loaders import TextLoader
# loads simple markdown or text file.
loader = TextLoader("./README.md")


from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
# loader = PyPDFLoader("./cat_and_dog_scene.pdf", extract_images = True)   
# pages = loader.load()


# print(pages, len(pages))
# loader = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
# pages = loader.load()
# print(pages[4].page_content)

# #vectorstore weaviate
# from langchain_community.vectorstores import Weaviate
# # vectorstore = Weaviate()
# # vectorstore.create_schema()
# # vectorstore.upload_documents(pages)

# text = "This is the text I would like to chunk up. It is the example text for this exercise"

# from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# text_splitter = CharacterTextSplitter(chunk_size = 35, chunk_overlap=4, separator='i', strip_whitespace=False)
# data = text_splitter.create_documents([text])

# # print(data)


# #PDF with Tables
# import os
# from unstructured.partition.pdf import partition_pdf
# from unstructured.staging.base import elements_to_json
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


# filename = "./sample.pdf"
# # Extracts the elements from the PDF
# elements = partition_pdf(
#     filename=filename,

#     # Unstructured Helpers
#     strategy="hi_res", 
#     infer_table_structure=True, 
#     model_name="yolox"
# )

# for idx in range(len(elements)):
#     print(f"Page {idx + 1} {elements[idx]} \n\n")
# #Table summarization and if query picks it, the table is sent to the LLM.


# #PDF with Images
# from typing import Any

# from pydantic import BaseModel
# from unstructured.partition.pdf import partition_pdf

# raw_pdf_elements = partition_pdf(
#     filename=filename,
    
#     # Using pdf format to find embedded image blocks
#     extract_images_in_pdf=True,
    
#     # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
#     # Titles are any sub-section of the document
#     infer_table_structure=True,
    
#     # Post processing to aggregate text once we have the title
#     chunking_strategy="by_title",
#     # Chunking params to aggregate text blocks
#     # Attempt to create a new chunk 3800 chars
#     # Attempt to keep chunks > 2000 chars
#     # Hard max on chunks
#     max_characters=2000,
#     new_after_n_chars=1800,
#     combine_text_under_n_chars=1000,
#     image_output_dir_path="./",
#     overlap_all = True,
# )

# svaes all images to a path. Later we generate summaries from the images and embed them

# print(raw_pdf_elements)
# for idx in range(len(raw_pdf_elements)):
#     print(f"Page {idx + 1} {raw_pdf_elements[idx]} \n\n")
# print(len(raw_pdf_elements))
