import os
import json
import tempfile
import streamlit as st
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Initialize Azure Services ---


# Set page configuration once at the top!
st.set_page_config(page_title="📄 PDF Chat Assistant", layout="wide")

# --- Helper Functions ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_users(filepath="users.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def save_users(users, filepath="users.json"):
    with open(filepath, "w") as f:
        json.dump(users, f)

# Helper functions to load and save user questions
def load_user_questions(filepath="user_questions.json"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def save_user_questions(user_questions, filepath="user_questions.json"):
    with open(filepath, "w") as f:
        json.dump(user_questions, f)

# --- Initialize Persistent Data ---
users_file = "users.json"
users = load_users(users_file)

# --- Initialize Session State ---
if 'users' not in st.session_state:
    st.session_state['users'] = users  # load persistent users into session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'user_questions' not in st.session_state:
    st.session_state['user_questions'] = load_user_questions()



# --- Authentication UI (visible when not logged in) ---
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
                # Ensure the user's questions list exists
                if username not in st.session_state['user_questions']:
                    st.session_state['user_questions'][username] = []
                st.sidebar.success(f"Welcome {username}!")
            else:
                st.sidebar.error("Invalid username or password.")

if st.session_state.get("logged_in"):
    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.sidebar.success("Logged out successfully!")
        st.sidebar.button("Login", key="login_btn")
        #set logged in to false
# else:
#     # When not logged in, show a Login button (or the login form) with a unique key.
#     if st.sidebar.button("Login", key="login_btn"):
#         st.sidebar.info("Please use the login form below to sign in.")


# --- Main App Content (only for logged in users) ---
if st.session_state.get("logged_in"):
    # Main Title and Caption
    st.title("📄 PDF Chat Assistant")
    st.caption("🚀 Upload multiple PDFs and get AI-powered insights with source citations")
    
    # --- Sidebar: Previous Questions (ChatGPT Style) ---
    username = st.session_state['username']
    user_qs = st.session_state['user_questions'].get(username, [])
    with st.sidebar.expander("💬 Your Previous Questions", expanded=True):
        if user_qs:
            for i, q in enumerate(user_qs[::-1], 1):  # Latest questions first
                st.write(f"**Q{i}:** {q}")
        else:
            st.write("No previous questions yet.")
    

    
    # --- File Uploader and Document Analysis ---
    uploaded_files = st.file_uploader("📂 Upload PDF documents", type="pdf", accept_multiple_files=True)
    question = st.text_input("💬 Enter your research question:", placeholder="What do you want to know from these documents?")
    
    # Placeholder for analysis progress
    progress_placeholder = st.sidebar.empty()

    if uploaded_files and question:
        # Save the research question in the user's history
        st.session_state['user_questions'][username].append(question)
        save_user_questions(st.session_state['user_questions'])
        # st.success("Question saved!")
        try:
            with st.spinner("🔄 Analyzing documents..."):
                progress_placeholder.info("Initializing analysis...")
                
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
                
                progress_placeholder.info("Documents processed. Creating embeddings...")
                
                # Create Vector Store and Retriever
                vector_store = FAISS.from_documents(all_docs, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                
                progress_placeholder.info("Embeddings created. Generating AI response...")
                
                # Define System Prompt
                system_prompt = """
                  You are an expert in text extraction and working on bills and invoices.
                  Your task is to process invoices and particulars and financial results with their corresponding amounts.
                  Particulars: Extract the description of the charges/services.
                  Financial_Amount: Extract the corresponding quarterly amount for each particular.
                  Also check for 'Total Amount', 'Net Amount' or 'Discount' if available; if not, they should default to 0.
                Follow these rules:
                1. Answer strictly based on provided documents.
                2. Cite sources using [source][page number] notation.
                3. If unsure, state clearly when information is not found.
                4. Keep answers concise and factual.
                
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
                
                # Execute RAG Chain
                response = rag_chain.invoke({"input": question})
                
                progress_placeholder.empty()
                
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
            progress_placeholder.empty()
            st.error(f"❌ Error during processing: {str(e)}")
    
    elif not uploaded_files:
        st.warning("⚠️ Please upload PDF documents to begin analysis.")
    elif not question:
        st.warning("⚠️ Please enter a research question to start your analysis.")
else:
    st.info("Please log in to use the app.")

