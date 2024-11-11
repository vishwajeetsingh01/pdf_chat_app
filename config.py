from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class process_pdf():

    def get_pdf_text(pdf_docs):
        text=""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text+= page.extract_text()
        return text
    
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        chunks = text_splitter.split_text(text)
        return chunks
    
    def get_vector_store(text_chunks):
        # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")


    def get_conversational_chain():

        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the data
        if the answer is not in provided context just say, "answer is not available in the context",
        don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        
        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        prompt = PromptTemplate(template = prompt_template, input_variables = {"context", "question"})
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain