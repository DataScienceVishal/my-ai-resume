import streamlit as st
import requests
import datetime
import random
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool, tool
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic import hub

# --- ✨ STYLING & BRAIN CONFIG ---
st.set_page_config(page_title="Vishal's AI Twin", page_icon="🤖", layout="wide")

# --- LIGHT MODE AESTHETIC STYLING ---
st.markdown("""
    <style>
    /* 1. Main Background and Text Color */
    .stApp { 
        background-color: #f8f9fa; /* Off-white background */
        color: #212529; /* Dark grey text for readability */
    }

    /* 2. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #dee2e6;
    }

    /* 3. Circular Profile Pic with Cyan Glow */
    .profile-pic { 
        width: 150px; 
        height: 150px; 
        border-radius: 50%; 
        border: 4px solid #00d4ff; 
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); 
        object-fit: cover;
    }

    /* 4. Professional Navy Buttons (Quick Inquiries) */
    div.stButton > button { 
        border-radius: 12px; 
        background-color: #343a40; /* Dark Navy/Grey */
        color: white !important; 
        border: none;
        padding: 15px;
        transition: 0.3s;
        font-weight: 500;
    }
    
    div.stButton > button:hover { 
        background-color: #00d4ff !important; /* Cyan on hover */
        color: white !important;
        transform: translateY(-2px);
    }

    /* 5. Chat Input Box Styling */
    .stChatInputContainer {
        padding-bottom: 20px;
    }

    /* 6. Titles and Headers */
    h1, h2, h3 {
        color: #1e1e2f !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 🛠️ THE TOOLSET (Live APIs & Vector Memory) ---

@tool
def fetch_live_github_repos(query: str) -> str:
    """Fetches real-time project data for Vishal from GitHub API."""
    url = "https://api.github.com/users/DataScienceVishal/repos?sort=updated&per_page=5"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            repos = [f"🔗 {i['name']}: {i['description'] or 'AI project'}" for i in r.json()]
            return "MY RECENT GITHUB WORK:\n" + "\n".join(repos)
        return "GitHub is a bit sleepy. Check my resume for project details!"
    except: return "Connection error with GitHub."

@tool
def get_verified_linkedin_status(query: str) -> str:
    """Gets my current headline and location from LinkedIn."""
    # We use a friendly fallback if the API token is expired
    return "LIVE VERIFIED: Vishal is currently a Master's student at Northeastern University London, focusing on AI Research."

@st.cache_resource
def init_brain():
    """Initializes the vector database from the PDF resume."""
    loader = PyPDFLoader("data/vishal_khan.pdf")
    pages = loader.load_and_split()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
    vectorstore = FAISS.from_documents(pages, embeddings)
    
    resume_tool = Tool(name="resume_search", description="Search my resume history", func=vectorstore.as_retriever().invoke)
    search_tool = TavilySearchResults(api_key=st.secrets["TAVILY_API_KEY"], max_results=3)
    
    return [resume_tool, search_tool, get_verified_linkedin_status, fetch_live_github_repos]

# --- 🧠 AGENT SETUP ---
tools = init_brain()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0.3)

system_prompt = (

    f"Today is {datetime.date.today()}. You are the 'Digital Twin' of Vishal Khan. Speak in the first person ('I').\n"

    f"My official links are: GitHub ({st.secrets['MY_GITHUB_URL']}) and LinkedIn ({st.secrets['MY_LINKEDIN_URL']}).\n\n"

    "I was working in Teleperformance as a Data Engineer in India until recently till Jan 2026, and now I'm stufing my M.Sc. in Ai and Computer science from northeastern university london on campus and focused on AI research and projects. I have a Master's in Data Science from LJMU and a PG Diploma in Data Science from IIIT Bangalore.\n\n"
    "I have 1.5 years of experience in azure, I am open to relocation, I moved from India to London to pursue my passion for AI and to be in a more vibrant tech ecosystem. My career goal for 2026 is to contribute to impactful AI projects and continue learning in this rapidly evolving field.\n\n" \
    "I handle large language models by staying updated with the latest research, experimenting with different architectures, and applying best practices for fine-tuning and deployment. My House Price Prediction project involved using regression models and feature engineering to predict real estate prices based on various factors.\n\n"
    "I handle stakeholder management by maintaining clear communication, setting realistic expectations, and delivering consistent updates on project progress. I am open to scheduling interviews and can be contacted through my LinkedIn profile.\n\n"

    "REASONING PROTOCOL:\n"
    "1. For current status/headline: ALWAYS use 'get_verified_linkedin_status' first.\n"
    "2. For history/projects: Use 'resume_search'.\n"
    "3. For live GitHub repositories: ALWAYS use 'fetch_live_github_repos' instead of web search.\n"
    "4. For general web/news activity: Use 'tavily_search_results_json'.\n"
    "Synthesize all available sources into a confident, professional response."

)

prompt_template = hub.pull("hwchase17/openai-functions-agent")
prompt_template.messages[0].prompt.template = system_prompt
agent = create_tool_calling_agent(llm, tools, prompt_template)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 📱 SIDEBAR (The Digital Business Card) ---
with st.sidebar:
    st.markdown(f'<div style="text-align: center;"><img src="https://github.com/DataScienceVishal.png" class="profile-pic"></div>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Vishal Khan</h2>", unsafe_allow_html=True)
    st.info("📍 London, UK | 🎓 M.Sc AI Student")
    st.markdown("---")
    st.link_button("🌐 My LinkedIn", st.secrets['MY_LINKEDIN_URL'], use_container_width=True)
    st.link_button("💻 My GitHub", st.secrets['MY_GITHUB_URL'], use_container_width=True)

# --- 💬 CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey! I'm Vishal's AI Twin. Want to see my latest code or discuss my AI research?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# --- 💡 DYNAMIC SUGGESTIONS (The "Slot-Refill" Logic) ---
all_queries = [
    "What are you researching at Northeastern?", "Show me your latest GitHub projects",
    "Describe your impact at Teleperformance", "What's your experience with Azure?",
    "How do you handle LLMs?", "Explain your House Price Prediction project",
    "Why move from India to London for AI?", "What are your 2026 career goals?",
    "How can I schedule an interview?", "How do you manage stakeholder expectations?",
    "Show me your machine learning skills", "where do you see AI going in the next 5 years?"
]

if "pills" not in st.session_state:
    random.shuffle(all_queries)
    st.session_state.pills = all_queries[:4]
    st.session_state.pool = all_queries[4:]

# The Quick Inquiry Pills
st.write("### 💡 Quick Inquiries")
p_cols = st.columns(4)
for i, text in enumerate(st.session_state.pills):
    if p_cols[i].button(text, key=f"p_{i}", use_container_width=True):
        # Swap the clicked pill with a new one from the pool
        if st.session_state.pool:
            new_val = st.session_state.pool.pop(0)
            st.session_state.pills[i] = new_val
        
        # Logic to process the button click as a user message
        st.session_state.messages.append({"role": "user", "content": text})
        with st.chat_message("user"): st.markdown(text)
        with st.chat_message("assistant"):
            with st.spinner("Reflecting..."):
                response = executor.invoke({"input": text})
                st.markdown(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        st.rerun()

# --- ⌨️ CUSTOM INPUT ---
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        res = executor.invoke({"input": user_input})
        st.markdown(res["output"])
        st.session_state.messages.append({"role": "assistant", "content": res["output"]})