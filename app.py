import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import io
from PyPDF2 import PdfReader
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Set environment variables from .env
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

# Function to extract text from a PDF
def get_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.read()
    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_reader = PdfReader(pdf_stream)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks with unique IDs
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    
    # Create a list of Document objects with IDs
    docs = [Document(page_content=chunk, metadata={"id": str(i)}) for i, chunk in enumerate(chunks)]
    return docs

# Function to set up the vectorstore with embeddings
def setup_vectorstore(docs):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    # Setting up embeddings and vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_name = 'qabot'
    
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key
    )
    
    return vectorstore, embeddings

# Function to set up the RAG chain
def setup_rag_chain(vectorstore, model_name="gemini-1.5-pro"):
    template = """Answer the question as detailed as possible from the provided context more that 500 words, make sure to provide all the details, 
    if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    {context}
    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.01)
    retriever = vectorstore.as_retriever()

    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# Function to handle the query and return an answer
def answer_question(chain, query):
    answer = chain.invoke(query)
    return answer

# Streamlit interface
def main():
    st.title("RAG Model for QA Bot")
    
    # File upload for PDF
    uploaded_file = st.file_uploader("Upload your PDF Files and Click on the Submit & Process", type="pdf")
    
    if uploaded_file:
        # Display PDF name
        with st.spinner("Processing..."):
        
            # st.write(f"Processing... {uploaded_file.name}...")
            
            # Extract text from the uploaded PDF
            pdf_text = get_pdf_text(uploaded_file)
            
            # Split the extracted text into chunks
            docs = get_text_chunks(pdf_text)
            # st.write(f"PDF split into {len(docs)} chunks.")
            
            # Set up the vectorstore automatically after the file is processed
            vectorstore, embeddings = setup_vectorstore(docs)
            # st.write("VectorStore setup complete and PDF data stored.")
            
            # Set up the RAG chain
            chain = setup_rag_chain(vectorstore)
            st.session_state['chain'] = chain
            # st.write("RAG chain setup complete and ready for queries.")
    
    # Once the chain is set up, we can start answering questions
    if 'chain' in st.session_state:
        query = st.text_input("Ask a question:")
        
        if query:
            answer = answer_question(st.session_state['chain'], query)
            st.write(f"Answer: {answer}")

# Run the Streamlit app
if __name__ == "__main__":
    main()