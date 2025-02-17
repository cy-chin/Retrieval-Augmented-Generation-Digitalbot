# Retrieval-Augmented-Generation-Digitalbot

This project demonstrates a simple implementation of Retrieval-Augmented Generation (RAG) using LangChain, Pinecone, and OpenAI to build a digitalbot capable of answering questions based on proprietary information. This approach enhances Large Language Models (LLM) by integrating real-time retrieval with text generation for more accurate responses.

### Prerequisite

---

The code leverages [Pincone](https://www.pinecone.io/) as the Knowledge Base, and [OpenAI](https://platform.openai.com/) as the LLM for text generation. You will need to generate a API Key to Pincone VectorStore and OpenAI for the code to function.

Alternative, feel free to explore other VectorStore solutions, such as [Milvus](https://milvus.io/) and [Chroma](https://www.trychroma.com/). Refer to [this reference](https://python.langchain.com/docs/how_to/local_llms/) for setting up and running a local LLM.

### Setup

---

1. Clone the repository to your local
2. Create a virtual environment
   ```
   > python -m venv ragvenv
   ```
3. Activate the virtual environment
   ```
   > .\ragvenv\Scripts\activate
   ```
4. Install the libraries in the requirements.txt
   ```
   > pip install -r requirements.txt
   ```
5. Replace the API key in*.env*
   ```
   OPENAI_API_KEY=[API KEY FROM OPENAI]
   PINECONE_API_KEY=[API KEY FROM PINCONE]
   ```
