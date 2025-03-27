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
              You are an expert in text extraction and working on bills and invoices.
                Your task is to process  invoices and particulars and financials reasults  with their corresponding amounts.
                Particulars: Extract the description of the charges/services.
                Financial_Amount: Extract the corresponding  quarterly amount for each particular
                Also check for 'Total Amount', 'Net Amount' or 'Discount' if available extract them; if not, they should default to 0.
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





# --- Authentication UI (only visible if not logged in) ---
if not st.session_state.get("logged_in"):
    auth_option = st.sidebar.radio("Authentication", ("Login", "Sign Up"))

    if auth_option == "Sign Up":
        st.sidebar.subheader("Create New Account")
        new_username = st.sidebar.text_input("Username", key="signup_username")
        new_password = st.sidebar.text_input("Password", type="password", key="signup_password")
        if st.sidebar.button("Sign Up"):
            if new_username and new_password:
                if new_username in st.session_state['users']:
                    st.sidebar.error("Username already exists!")
                else:
                    st.session_state['users'][new_username] = hash_password(new_password)
                    save_users(st.session_state['users'], users_file)
                    st.sidebar.success("Account created successfully! Please log in.")
            else:
                st.sidebar.error("Please provide both username and password.")

    if auth_option == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            stored_users = st.session_state['users']
            if username in stored_users and stored_users[username] == hash_password(password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                # Initialize the user's questions list if not already present
                if username not in st.session_state['user_questions']:
                    st.session_state['user_questions'][username] = []
                st.sidebar.success(f"Welcome {username}!")
            else:
                st.sidebar.error("Invalid username or password.")
                
# --- Main App Content (only for logged in users) ---
if st.session_state.get("logged_in"):
    # Add a logout button in the sidebar
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.sidebar.success("Logged out successfully!")
    
    # Main Title and Caption
    st.title("📄 PDF Chat Assistant")
    st.caption("🚀 Upload multiple PDFs and get AI-powered insights with source citations")
    
    # --- Sidebar: Previous Questions (ChatGPT Style) ---
    username = st.session_state['username']
    user_qs = st.session_state['user_questions'].get(username, [])
    with st.sidebar.expander("💬 Your Previous Questions", expanded=True):
        if user_qs:
            for i, q in enumerate(user_qs[::-1], 1):  # showing latest first
                st.write(f"**Q{i}:** {q}")
        else:
            st.write("No previous questions yet.")
    
    # File Uploader and Question Input
    uploaded_files = st.file_uploader("📂 Upload PDF documents", type="pdf", accept_multiple_files=True)
    question = st.text_input("💬 Enter your research question:", placeholder="What do you want to know from these documents?")
    
    # Placeholder for analysis progress (only shows during processing)
    progress_placeholder = st.sidebar.empty()

    if uploaded_files and question:
        # Save the question in session state for this user.
        st.session_state['user_questions'][username].append(question)
        
        try:
            with st.spinner("🔄 Analyzing documents..."):
                progress_placeholder.info("Initializing analysis...")
                # Process documents...
                # [Your document processing, embedding, and chain creation logic here]
                progress_placeholder.empty()
                st.subheader("📜 Research Findings")
                st.info("**Answer:** " + response['answer'])
                st.divider()
                st.subheader("📚 Source References")
                # [Display source references logic here]
        
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ Error during processing: {str(e)}")
    
    elif not uploaded_files:
        st.warning("⚠️ Please upload PDF documents to begin analysis.")
    elif not question:
        st.warning("⚠️ Please enter a research question to start your analysis.")
else:
    st.info("Please log in to use the app.")


