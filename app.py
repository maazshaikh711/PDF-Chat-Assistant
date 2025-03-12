import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Streamlit UI Config
st.set_page_config(page_title="📄 PDF Chat Assistant", layout="wide")
st.title("📄 PDF Chat Assistant")
st.caption("🚀 Upload multiple PDFs and get AI-powered insights with source citations")

# Sidebar UI Enhancements
st.sidebar.title("🔍 Analysis Progress")
st.sidebar.write("Monitor processing in real-time!")

# Initialize Azure Services
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding_model",
    openai_api_version="2024-05-01-preview",
    azure_endpoint="https://stock-agent.openai.azure.com/",
    api_key="7219267fcc1345cabcd25ac868c686c1"
)

llm = AzureChatOpenAI(
    azure_deployment="model-4o",
    openai_api_version="2024-05-01-preview",
    azure_endpoint="https://stock-agent.openai.azure.com/",
    api_key="7219267fcc1345cabcd25ac868c686c1",
    temperature=0.7,
    max_tokens=500
)

# File Uploader
uploaded_files = st.file_uploader("📂 Upload PDF documents", type="pdf", accept_multiple_files=True)
question = st.text_input("💬 Enter your research question:", placeholder="What do you want to know from these documents?")

if uploaded_files and question:
    try:
        with st.spinner("🔄 Analyzing documents..."):
            st.sidebar.progress(10)
            
            # Process documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )
            
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file.flush()
                    loader = PyPDFLoader(temp_file.name)
                    docs = loader.load()
                    splits = text_splitter.split_documents(docs)
                    all_docs.extend(splits)
            
            st.sidebar.progress(40)
            st.sidebar.success("✅ Documents Processed")
            
            # Create Vector Store
            st.sidebar.write("🧠 Creating Embeddings & Storing in FAISS DB...")
            vector_store = FAISS.from_documents(all_docs, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            st.sidebar.progress(70)
            
            # Define System Prompt
            system_prompt = """
            You are a professional research assistant. Analyze the following context to answer the question.
            Follow these rules:
            1. Answer strictly based on provided documents
            2. Cite sources using [source][page number] notation
            3. If unsure, state clearly when information is not found
            4. Keep answers concise and factual
            
            Context: {context}
            """
            
            # Create Prompt Template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Question: {input}")
            ])
            
            # Create Chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            st.sidebar.progress(90)
            st.sidebar.write("🤖 Generating AI Response...")
            
            # Execute RAG Chain
            response = rag_chain.invoke({"input": question})
            st.sidebar.progress(100)
            st.sidebar.success("🚀 Analysis Complete!")
            
            # Display Results
            st.subheader("📜 Research Findings")
            st.info(f"**Answer:** {response['answer']}")
            
            st.divider()
            st.subheader("📚 Source References")
            for i, doc in enumerate(response["context"]):
                source = doc.metadata.get("source", "Unknown document")
                page = doc.metadata.get("page", "N/A")
                with st.expander(f"📖 Source {i+1}: {source} (Page {page})"):
                    st.text(doc.page_content)
    
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")

elif not uploaded_files:
    st.warning("⚠️ Please upload PDF documents to begin analysis.")
elif not question:
    st.warning("⚠️ Please enter a research question to start your analysis.")