import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup Page (Mac Retina Display friendly)
st.set_page_config(page_title="Vishal's AI Assistant", page_icon="🤖")
st.title("👨‍💻 Vishal Khan: AI Portfolio Assistant")


# 2. Retrieve API Key from Backend Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing API Key! Please add GOOGLE_API_KEY to your secrets.toml or Cloud settings.")
    st.stop()

# 3. Cached Data Processing (Saves your Quota!)
@st.cache_resource
def initialize_vector_store(api_key):
    """Loads PDF and creates embeddings only once per session."""
    try:
        loader = PyPDFLoader("data/vishal_khan.pdf")
        pages = loader.load_and_split()
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key,
            task_type="retrieval_document"
        )
        return FAISS.from_documents(pages, embeddings)
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return None

if api_key:
    # Initialize or get from cache
    vectorstore = initialize_vector_store(api_key)
    
    if vectorstore:
        # 4. Setup RAG Chain (Using Gemini 2.5 Flash - Best Stability)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            temperature=0.3
        )
        
        system_prompt = (
            "You are Vishal Khan's professional AI assistant. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "Feel free to use the hyperlinks in the resume to provide more detailed answers."
            "Vishal is studing his masters in Northeastern University as an International Student from India. His previous studies or professional experience were in India."
            "Do not make up information. Only answer based on the provided context. "
            "Do not provide any strong weakness which can result in a negative impression. "
            "If you don't know the answer based on the resume, say you don't know. "
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Modern LangChain Chain Construction
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

        # 5. Chat Interface Logic
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        if user_input := st.chat_input("Ask me about Vishal's research or projects..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing resume..."):
                    try:
                        # Invoke the chain
                        response_dict = qa_chain.invoke({"input": user_input})
                        answer = response_dict["answer"]
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Quota error: Please wait 60 seconds and try again. ({e})")
else:
    st.info("👋 Please enter your Gemini API key in the sidebar to begin.")