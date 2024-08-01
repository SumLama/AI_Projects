import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context but dont provide provide the wrong answer
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    return response["output_text"]
   

user_info = {"name":"","phone_number":"","email":""}
def collect_user_info():
    if user_info['name'] == "":
        user_info['name'] = st.text_input("Enter your name:")

    if user_info['phone_number'] == "":
        user_info['phone_number'] = st.text_input("Enter your phone number:")    

    if user_info['email'] == "":
        user_info['email']  = st.text_input("Enter your email:")    

    if user_info['name'] and user_info['phone_number'] and user_info['email']:
        return "All information  collected. We will contact you shortly."
    else:
        return "Please provide all required information"

def main():
    st.set_page_config(page_title="PDF reader")
    st.header(" Your PDF Query Assistant")

    pdf_docs = st.file_uploader("Upload your PDF files and click on the submit button ", accept_multiple_files=True, type=["pdf"])
    if st.button("Submit",):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        else:
            st.warning("Please upload at least one PDF file.")
    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        if "call me" in user_question.lower():
            details = collect_user_info()
            st.write(details)
            
        else:
            response = user_input(user_question)
            st.write("Reply: ", response)
    
if __name__ == "__main__":
    main()
