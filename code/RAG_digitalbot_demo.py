import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Initialize Pinecone and LLM settings
index_name = 'gentingsp-financial2023'
namespace_name = 'gsp1'
embedding_model = 'text-embedding-ada-002'
llm_model = "gpt-3.5-turbo"
llm_temperature = 0.7

def init():
    # Load environment variables
    load_dotenv()

    # intialize embedding model
    embeddings = OpenAIEmbeddings(
        model=embedding_model
    )

    #initialize pincone vectorstore DB connection
    pc_vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings, namespace=namespace_name
    )

    #initialize openAi connection
    llm = ChatOpenAI(
        model=llm_model,
        temperature=llm_temperature
    )

    # Custom RAG Prompt
    custom_rag_prompt = PromptTemplate(
        template=(
            "You are an assistant for question-answering tasks.\n\n"
            "Use the following pieces of retrieved context to answer the question.\n\n"
            "If the answer cannot be derived from the context, just say the information is not available in my Knowledge Base.\n\n"
            "Use not more than 10 sentences and keep the answer concise.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"]
    )

    #initialize retrieval chain
    rqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=pc_vector_store.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.1, "k": 6}
        ),
        chain_type_kwargs={"prompt": custom_rag_prompt}
    )

    return rqa

def startDigitalBot(qa):
    # Streamlit App
    st.title("RAG Digital Chatbot - DEMO")
    st.subheader(">> powered by Langchain|Pinecone|OpenAi <<")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=f"Welcome to Digital Chatbot for KB-{index_name}.")]

    # User input box
    user_input = st.text_input("Ask your question:", "")

    # If user submits a question
    if user_input:
        # Append the user's message to the session state
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Get the response from the chat model
        with st.spinner("Thinking..."):
            try:
                response = qa.invoke({"query": user_input}).get("result", "No answer available.")
                st.session_state.messages.append(AIMessage(content=response))
            except Exception as e:
                st.session_state.messages.append(AIMessage(content=f"Error: {e}"))

    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.write(f"**You:** {message.content}")
        elif isinstance(message, AIMessage):
            st.write(f"**Bot:** {message.content}")
        elif isinstance(message, SystemMessage):
            st.write(f"**System:** {message.content}")


if __name__ == "__main__":
    startDigitalBot(init())

    