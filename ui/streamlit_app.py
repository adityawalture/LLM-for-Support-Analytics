import streamlit as st
import requests
import os

# Set page config
st.set_page_config(
    page_title="Support Ticket Analytics Dashboard",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Inject Custom CSS for Clean Minimalist Aesthetics
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global Font Override */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Dashboard Header Title */
.main-title {
    font-weight: 700;
    font-size: 2.2rem;
    margin-bottom: 0.1rem;
}

.sub-title {
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

/* Anomaly alert card styling */
.anomaly-card {
    background: rgba(239, 68, 68, 0.05);
    border-left: 3px solid #ef4444;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 8px;
}

.anomaly-title {
    font-weight: 600;
    color: #ef4444;
    font-size: 0.95rem;
    margin-bottom: 2px;
}

.anomaly-reason {
    font-size: 0.85rem;
}

/* Pulsing Status Dot */
.status-dot-green {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
}

.status-dot-red {
    width: 8px;
    height: 8px;
    background-color: #ef4444;
    border-radius: 50%;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# Helper function to check health status
def check_backend_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if r.status_code == 200 and r.json().get("status") == "running":
            return True
    except Exception:
        pass
    return False

# Helper function to retrieve summary stats
def get_summary_stats():
    try:
        r = requests.get(f"{API_BASE_URL}/summary")
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Helper function to query backend
def post_nlp_query(question):
    try:
        r = requests.post(f"{API_BASE_URL}/query", json={"question": question})
        if r.status_code == 200:
            return r.json().get("answer")
        else:
            return f"Error: API returned status code {r.status_code} ({r.text})"
    except Exception as e:
        return f"Connection Error: Could not connect to API at {API_BASE_URL} ({str(e)})"

# Helper function to retrieve anomalies
def get_anomalies():
    try:
        r = requests.get(f"{API_BASE_URL}/anomalies")
        if r.status_code == 200:
            return r.json()
    except Exception:
        return []
    return []

# Sidebar Navigation and Connection Status
with st.sidebar:
    st.markdown("## 🎫 System Diagnostics")
    
    # Live Connection Status Check
    is_online = check_backend_health()
    if is_online:
        st.markdown('<div class="status-dot-green"></div><span style="color:#10b981;font-weight:600;">API Backend Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-dot-red"></div><span style="color:#ef4444;font-weight:600;">API Backend Offline</span>', unsafe_allow_html=True)
        st.warning("FastAPI backend is offline. Run `uvicorn app.main:app --reload` to start it.")

    st.markdown("---")
    st.markdown("### 🤖 LLM Strategy")
    st.info("The system uses an **LLM-guided Pandas execution pattern** which translates natural language questions into structured queries, performs calculation in Pandas, and generates descriptive summaries. Keeps responses deterministic and hallucination-free!")
    
    # Display config parameters
    st.markdown("### 🔌 Endpoints")
    st.code(f"API Base: {API_BASE_URL}\nHealth: {API_BASE_URL}/health\nQuery: {API_BASE_URL}/query\nAnomalies: {API_BASE_URL}/anomalies\nSummary: {API_BASE_URL}/summary", language="yaml")

# Header Section
st.markdown('<h1 class="main-title">TicketAnalytics AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Support Ticket Querying & SLA Anomaly Engine</p>', unsafe_allow_html=True)

# Fetch stats summary
stats = get_summary_stats()
if stats:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Tickets", stats['total_tickets'])
    col_m2.metric("Open Tickets", stats['open_tickets'])
    col_m3.metric("Resolved Tickets", stats['resolved_tickets'])
    col_m4.metric("Avg Rating", f"{stats['average_customer_rating']} ⭐" if stats['average_customer_rating'] else "N/A")
else:
    st.info("Ingesting ticket details. Check if the API backend is running to load metrics.")

st.markdown("---")

# Create Two-Column Layout for Queries and Anomalies
col_query, col_anomaly = st.columns([1.2, 0.8])

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with col_query:
    st.markdown("### 💬 Ask Questions")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant", avatar="🤖"):
            st.write(chat["answer"])
            
    # Quick suggestion presets
    st.markdown("<br>##### Quick Suggestions:", unsafe_allow_html=True)
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        q_open = st.button("How many tickets are currently open?", use_container_width=True)
        q_lowest = st.button("Which agent has the lowest customer rating?", use_container_width=True)
    with preset_col2:
        q_sla = st.button("Show Critical tickets unresolved after 12h.", use_container_width=True)
        q_most = st.button("Which agent resolved most tickets this month?", use_container_width=True)

    # Input handling
    query_text = None
    if q_open:
        query_text = "How many tickets are currently open?"
    elif q_lowest:
        query_text = "Which agent has the lowest average customer rating?"
    elif q_sla:
        query_text = "Show me all Critical tickets not resolved within 12 hours."
    elif q_most:
        query_text = "Which agent resolved the most tickets this month?"

    chat_input_val = st.chat_input("Ask a question about support tickets...")
    if chat_input_val:
        query_text = chat_input_val

    if query_text:
        if not is_online:
            st.error("Cannot query. The FastAPI backend is offline.")
        else:
            with st.spinner("Analyzing dataset & formulating answer..."):
                answer = post_nlp_query(query_text)
                
            # Save to history
            st.session_state.chat_history.append({"question": query_text, "answer": answer})
            # Rerun to update state
            st.rerun()

with col_anomaly:
    st.markdown("### ⚠️ Anomaly Dashboard")
    
    if not is_online:
        st.warning("Connect the API to list system anomalies.")
    else:
        anomalies = get_anomalies()
        if len(anomalies) == 0:
            st.success("✅ No anomalies detected in support operations.")
        else:
            st.markdown(f"Detected **{len(anomalies)}** operations anomalies:")
            
            # Search filter for anomalies
            search_query = st.text_input("Search anomalies:", placeholder="Filter by Ticket ID or reason...").strip().lower()
            
            # Simple Scrollable Container
            with st.container(height=400):
                for item in anomalies:
                    ticket_id = item['ticket_id']
                    reason = item['reason']
                    
                    # Filter by search
                    if search_query and (search_query not in ticket_id.lower() and search_query not in reason.lower()):
                        continue
                        
                    st.markdown(f"""
                    <div class="anomaly-card">
                        <div class="anomaly-title">🎫 {ticket_id}</div>
                        <div class="anomaly-reason">{reason}</div>
                    </div>
                    """, unsafe_allow_html=True)
