import os
from dotenv import load_dotenv
import streamlit as st      # this helps to create UI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from config import process_pdf


load_dotenv(override=True)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dengerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = process_pdf.get_conversational_chain()


    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("PDF Chat Bot", page_icon = ":scroll:")
    st.header("PDF📖 - Chat Bot 🤖")

    user_question = st.text_input("Ask Question from the PDF Files uploaded .. ✍️📖")

    if user_question:
        user_input(user_question)

    with st.sidebar:

        # st.image("logo.jpg")
        # st.write("---")

        st.title(" PDF File's section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit Button ", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."): # user friendly message.
                raw_text = process_pdf.get_pdf_text(pdf_docs) # get the pdf text
                text_chunks = process_pdf.get_text_chunks(raw_text) # get the text chunks
                process_pdf.get_vector_store(text_chunks) # create vector store
                st.success("Done")


if __name__ == "__main__":
    main()
