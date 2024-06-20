import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import TransformerEmbeddings
from langchain.chains import load_qa_chain
from langchain.prompts import PromptTemplate
import os

def get_pdf_text(pdf_files):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_vector_store(text_chunks, embeddings):
    """Create and save a vector store."""
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Set up the QA conversational chain."""
    prompt_template = PromptTemplate(
        template="""Context: {context}\nQuestion: {question}\nAnswer:""",
        input_variables=["context", "question"]
    )
    model = TransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    chain = load_qa_chain(model, prompt=prompt_template)
    return chain

def main():
    """Main function to run the Streamlit app."""
    st.title("PDF Conversational Interface")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Process PDFs"):
        if uploaded_files:
            with st.spinner('Extracting and processing text from PDFs...'):
                extracted_text = get_pdf_text(uploaded_files)
                text_chunks = extracted_text.split('.')  # Simple split by sentence
                embeddings = TransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                get_vector_store(text_chunks, embeddings)
                st.success("PDFs processed and data indexed.")
        else:
            st.error("Please upload some PDF files.")

    user_question = st.text_input("Ask a question:")
    if user_question:
        embeddings = TransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = FAISS.load_local("faiss_index", embeddings)
        docs = vector_store.similarity_search(user_question, top_k=5)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Answer:", response)

if __name__ == "__main__":
    main()
