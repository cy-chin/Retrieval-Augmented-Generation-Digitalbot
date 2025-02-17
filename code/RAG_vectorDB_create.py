from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

import pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from dotenv import load_dotenv
import os
from tqdm import tqdm

# Initialize Pinecone and splitter parameters
index_name = 'gentingsp-financial2023'
namespace_name = 'gsp1'
dimension = 1536
metric = "cosine"
embedding_model = 'text-embedding-ada-002'

chunk_size = 2000
chunk_overlap = 500


def init():
    # Load environment variables
    load_dotenv()

    # intialize embedding model
    embeddings = OpenAIEmbeddings(
        model=embedding_model
    )

    pc = Pinecone(api_key = os.environ["PINECONE_API_KEY"])
    
    #loop through pincone indexes to retrieve the pincone vectorstore DB index. 
    # if index_name is first time, create the index, with a default record (so that namespace can be created)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 

        #insert a dummy record so that namespace can be created with namespace_name
        pc_index = pc.Index(index_name)
        pc_index.upsert(
            vectors=[{"id":"def001", "values":[0.0000001 for _ in range(dimension)]},],
            namespace=namespace_name
        )
        print(f"Created Index {index_name} with a namespace {namespace_name}")
        
    else:
        print(f"Index {index_name} already exist")
        pc_index = pc.Index(index_name)

    
    return embeddings, pc_index, splitDocuments(chunk_size, chunk_overlap)

def splitDocuments(csize, coverlap):

    DATA_PATH = "./data"
    loader = PDFPlumberLoader(file_path=f"{DATA_PATH}/GentingSingapore_AnnualReport_2023.pdf")
    docs = loader.load()  #loader will load the documents as list of Langchain document
    print(f"PDF file loaded")

    #initiate text_splitter, and split the langchain document (from PDF load) into chunks
    text_splitter = RecursiveCharacterTextSplitter(  
        chunk_size=csize,
        chunk_overlap=coverlap,
        length_function=len,
        is_separator_regex=False,)

    split_docs = text_splitter.split_documents(docs)

    return split_docs

    
def createVectorDB():

    embeddings, pc_index, split_docs = init()

    # Insert chunks into the Pinecone index
    for i, doc in tqdm(enumerate(split_docs), desc="Processing chunks", total=len(split_docs)):
        # Generate the embedding for the chunk
        embedding = embeddings.embed_query(doc.page_content)

        # Create metadata for the chunk
        metadata = {
            "text": doc.page_content,  
            "source": doc.metadata.get("source", "unknown"),  
            "chunk_id": f"chunk_{i}", 
        }

        # Upsert the vector into Pinecone
        pc_index.upsert(
            vectors=[{
                "id": f"chunk_{i}",
                "values": embedding,
                "metadata": metadata,
            }],
            namespace=namespace_name,
        )

    print("Chunks successfully uploaded to Pinecone!")

if __name__ == "__main__":
    createVectorDB()
