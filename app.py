__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Your existing imports continue right below here...
import streamlit as st
import os
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="RAG Q&A System", layout="centered")
st.title("📚 RAG-Based Research Paper Q&A")

# 1. Grab OpenAI API Key safely from environment or user input
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize Embedding Model
    openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. File Uploading Widget
    uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

    if uploaded_file:
        # Save file locally to process it with PyPDFLoader
        with open("temp_paper.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # We cache the processing so it doesn't re-index on every message click
        @st.cache_resource
        def initialize_rag(file_path):
            # Parse and Chunk
            loader = PyPDFLoader(file_path)
            doc_pages = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=300)
            doc_chunks = splitter.split_documents(doc_pages)
            
            # Create transient in-memory Chroma instance for the app session
            vector_db = Chroma.from_documents(
                documents=doc_chunks,
                embedding=openai_embed_model,
                collection_metadata={"hnsw:space": "cosine"}
            )
            return vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        with st.spinner("Parsing and indexing document..."):
            retriever = initialize_rag("temp_paper.pdf")
            st.success("Document successfully indexed!")

        # 3. Handle Prompt and LCEL Chain setup
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_prompt = """
        you are an assistant specialized in question answerinng and translation.
        your task is to respond to the given question using only the information provided in the retrieved context.

        Instructions:
        - If the answer is not present in the context, clearly state:"I don't know based on the given Context."
        - Do not invent or assume any information
        - write the answer in clear, simple language with correct grammer.
        - Make the response detailed, structured, and easy to understand.

        Question:
        {question}

        context:
        {context}

        Answer:
        """
        rag_prompt_template = ChatPromptTemplate.from_template(rag_prompt)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt_template
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )

        # 4. Streamlit Chat UI Session
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("Ask something about the document..."):
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("assistant"):
                with st.spinner("Searching context..."):
                    result = rag_chain.invoke(query)
                    response = result.content
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please provide your OpenAI API Key to start.")