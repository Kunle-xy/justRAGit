# Retrieval-Augmented Generation (RAG) Project

The Retrieval-Augmented Generation (RAG) Project is designed to transform how we interact with document data, offering a streamlined approach to analyze, vectorize, and comprehend files using state-of-the-art technologies. At the heart of RAG is Weaviate, an AI-powered vector database that facilitates efficient document vectorization. The project leverages the Langchain framework for creating robust data pipelines and Streamlit for crafting interactive user interfaces.


## Key Features

- **PDF Upload**: Securely upload PDF documents to be processed.
- **Text Extraction**: Utilize advanced algorithms to extract text from PDFs, breaking down content into manageable segments.
- **Chunk Storage**: Efficiently store extracted text chunks in Weaviate, ensuring quick retrieval and organization.
- **Embeddings Retrieval**: Generate and retrieve document embeddings, enabling deep semantic search and analysis.
- **LLM Integration**: Seamlessly integrate with Large Language Models for enhanced comprehension and generation tasks.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed on your machine. This project relies on several advanced Python libraries, including Langchain, Weaviate, and Streamlit, to provide a comprehensive document analysis and vectorization solution.

### Installation

1. **Clone the Repository**

    Start by cloning the RAG project repository to your local machine:

    ```bash
    git clone <repository-url>
    ```

2. **Install Dependencies**

    Navigate to the project directory and install the required Python libraries:

    ```bash
    cd path/to/rag-proect[justRagit]
    pip install -r requirements.txt
    ```

3. ### Usage: Indexing

Once installation is complete, you're ready to run the main application:

```bash
python src/main.py --pdf_file="path/to/your/document.pdf"
```
4. ### Usage: Embedding and Retrieval
   
Retrive text data based on given query.
Uses gpt-3.5-turbo-0613 model to generate answer
index two sample files from directory named 'pdf'
returns the answers based on top 5 

### DEMO
First step: [![Watch the video](https://img.youtube.com/vi/JyvfjnMqRW4/hqdefault.jpg)](https://www.youtube.com/watch?v=JyvfjnMqRW4)

Second step: https://www.youtube.com/watch?v=HD_PS3HMkCk

