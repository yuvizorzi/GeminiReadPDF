import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from langchain.text_splitter import RecursiveTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

#def custom_text_splitter(text, chunk_size = 4000, overlap = 1000):
    #chunks = []
    #start = 0
   # while start < len(text):
        #end = start + chunk_size
       # if end + overlap < len(text):
           # end += overlap
        #chunks.append(text[start:end])
        #start += chunk_size - overlap # Adjust start for overlap
    #return chunks

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide a detailed answer. If the answer is not provided in the context, just say 'Answer is not available in the context'. Do not provide wrong answer. \n\n
    Context: "\n{context}\n"
    Question: "\n{question}\n"
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response)

#from langchain_community.vectorstores import FAISS

#def main():
   # st.set_page_config(page_title="Chat with Multiple PDF", layout="wide")
    #st.header("Chat with Multiple PDF using Gemini💁")
   # user_question = st.text_input("Ask a Question from the PDF Files")

   # if user_question:
       # user_input(user_question)

  #  with st.sidebar:
       # st.title("Menu")
       # pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
       # if st.button("Submit & Process"):
           # if pdf_docs is None or len(pdf_docs) == 0:
               # st.warning("Please upload at least one PDF file to proceed.")
          #  else:
               # with st.spinner("Processing..."):
                  #  raw_text = get_pdf_text(pdf_docs)
                   # if raw_text:
                      #  text_chunks = custom_text_splitter(raw_text)
                      #  get_vector_store(text_chunks)
                      #  st.success("Processing complete!")
                   # else:
                      #  st.error("No text could be extracted from the uploaded PDF files. Please check your files.")

import streamlit as st
import logging

logging.basicConfig(level=logging.DEBUG)

try:
    if __name__ == "__main__":
        logging.info("Starting the Streamlit app")
        main()

except Exception as e:
    logging.error("An error occurred: ", exc_info=True)
    st.error(f"An error occurred: {e}")



#def main():
    #st.set_page_config(page_title="Chat with Multiple PDF", layout="wide")
    #st.header("Chat with Multiple PDF using Gemini💁")

    # Input for user questions
    #user_question = st.text_input("Ask a Question from the PDF Files")

    # Handle user questions if available
    #if user_question:
       # user_input(user_question)

    #with st.sidebar:
        #st.title("Menu")
        # Uploading PDF files
        #pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        #if st.button("Submit & Process"):
            #if pdf_docs is None or len(pdf_docs) == 0:
               # st.warning("Please upload at least one PDF file to proceed.")
            #else:
                #with st.spinner("Processing..."):
                    # Extract text from PDF
                   # raw_text = get_pdf_text(pdf_docs)
                    #if raw_text:  # Ensure there is text extracted
                   #     # Split and store the text for processing
                        #text_chunks = custom_text_splitter(raw_text)
                       # get_vector_store(text_chunks)
                       # st.success("Processing complete!")
                    #else:
                        #st.error("No text could be extracted from the uploaded PDF files. Please check your files.")

def main():
    st.set_page_config(page_title="Chat with Multiple PDF", layout="wide")
    st.header("Chat with Multiple PDF using Gemini💁")
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = custom_text_splitter(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

                
import streamlit as st
import logging

logging.basicConfig(level=logging.DEBUG)

try:
    # Your existing setup code here

    if __name__ == "__main__":
        logging.info("Starting the Streamlit app")
        main()

except Exception as e:
    logging.error("An error occurred: ", exc_info=True)
    st.error(f"An error occurred: {e}")

import pickle

def safe_load_pickle(filename, allow_dangerous_deserialization=False):
    if not allow_dangerous_deserialization:
        raise ValueError("Dangerous deserialization is not allowed. Make sure you trust the source before enabling it.")

    with open(filename, 'rb') as file:
        return pickle.load(file)

# Usage
try:
    # Only set to True if you are sure about the safety of the pickle file.
    my_data = safe_load_pickle("path_to_your_file.pkl", allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Failed to load pickle file: {e}")
