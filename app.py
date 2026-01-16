import streamlit as st
import asyncio
import os
import pandas as pd
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv

from moderation_core import (
    TextAnalysisAgent, 
    ImageRecognitionAgent, 
    CulturalContextAgent, 
    LegalComplianceAgent, 
    ArbitrationEngine, 
    MonitoringSystem, 
    ModerationContext
)
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Multi-Agent Guardian",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #f0f2f6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        background-color: #ff4b4b;
        color: white;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #464b5c;
    }
    .agent-card {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# State Management
if "monitor" not in st.session_state:
    st.session_state.monitor = MonitoringSystem()
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Configuration
with st.sidebar:
    st.title("Configuration")
    
    # api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password") # Removed for security
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY missing! Please add it to your .env file.")
        st.stop()
        
    model_name = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    
    st.markdown("---")
    st.subheader("System Status")
    st.metric("Total Processed", st.session_state.monitor.metrics["total_processed"])
    st.metric("Hard Limit Hits", st.session_state.monitor.metrics["hard_limit_hits"])

# Main Interface
st.markdown("<h1 class='main-header'>🛡️ Multi-Agent Content Guardian</h1>", unsafe_allow_html=True)
st.markdown("### Simulating a decentralized moderation committee")

col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("Content Input")
    
    user_region = st.selectbox("User Jurisdiction", ["global", "fr", "us", "eu", "sg", "jp", "in"])
    
    post_text = st.text_area("Post Content", height=150, placeholder="Type the content to moderate here...")
    
    st.markdown("#### Image Context (Simulated)")
    image_option = st.radio("Select Image Context:", ["None", "Safe Image", "Contains Nudity", "Contains Violence/Weapons"], horizontal=True)
    
    uploaded_file = st.file_uploader("Or upload an image (Mock analysis)", type=["png", "jpg", "jpeg"])
    
    # Determine context path
    image_path_context = None
    if image_option == "Contains Nudity":
        image_path_context = "image_with_nudity.jpg"
    elif image_option == "Contains Violence/Weapons":
        image_path_context = "image_with_gun_violence.jpg"
    elif uploaded_file:
         # In a real app, we'd save this. Here, we just use the name for mock analysis
         image_path_context = uploaded_file.name
    
    run_btn = st.button("Analyze Content")

async def run_moderation(text, img_path, region, key, model):
    # Initialize LLM
    try:
        llm = ChatGroq(api_key=key, model_name=model, temperature=0)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

    # Create Agents
    text_agent = TextAnalysisAgent("Text Analysis", llm, jurisdiction=region)
    image_agent = ImageRecognitionAgent("Image Recon", llm) # Global jurisdiction for vision usually
    cultural_agent = CulturalContextAgent("Cultural Context", llm, jurisdiction=region)
    legal_agent = LegalComplianceAgent("Legal Compliance", llm, jurisdiction=region)
    
    arbitrator = ArbitrationEngine()
    
    # Context
    context = ModerationContext(
        post_id=f"post_{datetime.now().strftime('%H%M%S')}",
        text=text,
        image_path=img_path,
        user_region=region
    )
    
    # Run Agents
    with st.status("Agents are deliberating...", expanded=True) as status:
        st.write("Constructing agent committee...")
        
        # We can run them concurrently
        tasks = [
            text_agent.moderate(context),
            image_agent.moderate(context),
            cultural_agent.moderate(context),
            legal_agent.moderate(context)
        ]
        
        st.write("Analyzing text and visual signals...")
        results = await asyncio.gather(*tasks)
        
        # Store results
        agents = [text_agent, image_agent, cultural_agent, legal_agent]
        for agent, result in zip(agents, results):
            context.agent_decisions[agent.name] = result
            st.write(f"{agent.name} finished.")

        st.write("Arbitrating final decision...")
        context.final_decision = await arbitrator.resolve(context)
        
        st.session_state.monitor.log_decision(context)
        
        # Log to history
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "post_id": context.post_id,
            "text": context.text,
            "region": context.user_region,
            "final_decision": context.final_decision,
            "conflicts": ", ".join(context.conflicts) if context.conflicts else "None",
            "text_agent_score": context.agent_decisions.get("Text Analysis", {}).get("confidence", 0),
            "image_agent_score": context.agent_decisions.get("Image Recon", {}).get("confidence", 0)
        })
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)
        
    return context

if run_btn:
    if not post_text and not image_path_context:
        st.warning("Please provide text or image content.")
    else:
        with col_result:
            result_context = asyncio.run(run_moderation(post_text, image_path_context, user_region, api_key, model_name))
            
            if result_context:
                st.subheader("Final Verdict")
                
                decision = result_context.final_decision.upper()
                color = "green" if decision == "ALLOW" else "red" if decision == "BLOCK" else "orange"
                
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h1 style="margin:0;">{decision}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                if result_context.conflicts:
                    st.warning(f"Conflicts Detected: {', '.join(result_context.conflicts)}")
                
                st.markdown("### Agent Breakdowns")
                
                for agent_name, decision_data in result_context.agent_decisions.items():
                    agent_decision = decision_data.get("decision", "N/A").upper()
                    agent_color = "#4caf50" if agent_decision == "ALLOW" else "#f44336" if agent_decision == "BLOCK" else "#ff9800"
                    
                    with st.expander(f"{agent_name}  —  {agent_decision}", expanded=True):
                        st.markdown(f"**Decision:** <span style='color:{agent_color}'>{agent_decision}</span>", unsafe_allow_html=True)
                        st.write(f"**Confidence:** {decision_data.get('confidence', 0.0):.2f}")
                        st.write(f"**Reasoning:** {decision_data.get('reason', 'None')}")
                        
                        # Show evidence log for this agent
                        agent_logs = [l for l in result_context.evidence_log if l['agent'] == agent_name]
                        if agent_logs:
                            st.markdown("**Evidence Log:**")
                            for log in agent_logs:
                                st.code(f"{log['evidence']} (Severity: {log['severity']})")

    # History and Export Section
    st.markdown("---")
    st.subheader("Session History & Audit Log")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Audit Log (CSV)",
            data=csv,
            file_name=f"moderation_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No analyses performed in this session yet.")
