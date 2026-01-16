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
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY missing! Please add it to your .env file.")
        st.stop()
        
    model_name = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    
    st.markdown("---")
    st.subheader("System Status")
    st.metric("Total Processed", st.session_state.monitor.metrics["total_processed"])
    st.metric("Hard Limit Hits", st.session_state.monitor.metrics["hard_limit_hits"])

# Core Logic
async def run_moderation(text, img_path, region, key, model, visualize=True):
    # Initialize LLM
    try:
        llm = ChatGroq(api_key=key, model_name=model, temperature=0)
    except Exception as e:
        if visualize: st.error(f"Failed to initialize LLM: {e}")
        return None

    # Create Agents
    text_agent = TextAnalysisAgent("Text Analysis", llm, jurisdiction=region)
    image_agent = ImageRecognitionAgent("Image Recon", llm) # Global jurisdiction for vision usually
    cultural_agent = CulturalContextAgent("Cultural Context", llm, jurisdiction=region)
    legal_agent = LegalComplianceAgent("Legal Compliance", llm, jurisdiction=region)
    
    arbitrator = ArbitrationEngine()
    
    # Context
    context = ModerationContext(
        post_id=f"post_{datetime.now().strftime('%H%M%S%f')}", # Microseconds for batch uniqueness
        text=text,
        image_path=img_path,
        user_region=region
    )
    
    # Run Agents
    tasks = [
        text_agent.moderate(context),
        image_agent.moderate(context),
        cultural_agent.moderate(context),
        legal_agent.moderate(context)
    ]
    
    if visualize:
        with st.status("Agents are deliberating...", expanded=True) as status:
            st.write("Constructing agent committee...")
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
    else:
        # Silent execution for batch
        results = await asyncio.gather(*tasks)
        agents = [text_agent, image_agent, cultural_agent, legal_agent]
        for agent, result in zip(agents, results):
            context.agent_decisions[agent.name] = result
        
        context.final_decision = await arbitrator.resolve(context)
        st.session_state.monitor.log_decision(context)
        
    return context

# Main Interface
st.markdown("<h1 class='main-header'>🛡️ Multi-Agent Content Guardian</h1>", unsafe_allow_html=True)

tab_single, tab_batch = st.tabs(["Single Analysis", "Batch Analysis"])

# --- TAB 1: SINGLE ANALYSIS ---
with tab_single:
    st.markdown("### Simulating a decentralized moderation committee")
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Content Input")
        user_region = st.selectbox("User Jurisdiction", ["global", "fr", "us", "eu", "sg", "jp", "in"])
        post_text = st.text_area("Post Content", height=150, placeholder="Type the content to moderate here...")
        
        st.markdown("#### Image Context (Simulated)")
        image_option = st.radio("Select Image Context:", ["None", "Safe Image", "Contains Nudity", "Contains Violence/Weapons"], horizontal=True)
        uploaded_file = st.file_uploader("Or upload an image (Mock analysis)", type=["png", "jpg", "jpeg"])
        
        image_path_context = None
        if image_option == "Contains Nudity": image_path_context = "image_with_nudity.jpg"
        elif image_option == "Contains Violence/Weapons": image_path_context = "image_with_gun_violence.jpg"
        elif uploaded_file: image_path_context = uploaded_file.name
        
        run_btn = st.button("Analyze Content", key="btn_single")

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
                            
                            agent_logs = [l for l in result_context.evidence_log if l['agent'] == agent_name]
                            if agent_logs:
                                st.markdown("**Evidence Log:**")
                                for log in agent_logs:
                                    st.code(f"{log['evidence']} (Severity: {log['severity']})")

    # History Section (Global for single analysis)
    st.markdown("---")
    st.subheader("Session History & Audit Log (Single Mode)")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History (CSV)", csv, f"moderation_history_{datetime.now().strftime('%H%M%S')}.csv", "text/csv")
    else:
        st.info("No single analyses performed yet.")

# --- TAB 2: BATCH ANALYSIS ---
with tab_batch:
    st.header("Batch Processor")
    st.markdown("Upload a CSV file with at least a **'text'** column. Optional columns: `'image_path'`, `'region'`.")
    
    uploaded_batch = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    
    if uploaded_batch:
        df_batch = pd.read_csv(uploaded_batch)
        st.dataframe(df_batch.head(), use_container_width=True)
        
        if "text" not in df_batch.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            if st.button("Process Batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                total = len(df_batch)
                
                for index, row in df_batch.iterrows():
                    status_text.text(f"Processing row {index + 1}/{total}...")
                    
                    text_input = str(row["text"])
                    img_input = str(row.get("image_path", "")) if pd.notna(row.get("image_path")) else None
                    region_input = str(row.get("region", "global")) # Default to global if missing
                    
                    # Run Analysis (Silent)
                    ctx = asyncio.run(run_moderation(text_input, img_input, region_input, api_key, model_name, visualize=False))
                    
                    if ctx:
                        results.append({
                            "original_text": text_input,
                            "final_decision": ctx.final_decision,
                            "conflicts": str(ctx.conflicts),
                            "text_agent_decision": ctx.agent_decisions.get("Text Analysis", {}).get("decision"),
                            "text_agent_reason": ctx.agent_decisions.get("Text Analysis", {}).get("reason"),
                            "image_agent_decision": ctx.agent_decisions.get("Image Recon", {}).get("decision"),
                            "legal_agent_decision": ctx.agent_decisions.get("Legal Compliance", {}).get("decision"),
                            "cultural_agent_decision": ctx.agent_decisions.get("Cultural Context", {}).get("decision"),
                            "processed_at": datetime.now().isoformat()
                        })
                    
                    progress_bar.progress((index + 1) / total)
                
                status_text.text("Batch processing complete!")
                
                # Show Results
                result_df = pd.DataFrame(results)
                st.subheader("Results Preview")
                st.dataframe(result_df, use_container_width=True)
                
                # Download
                csv_batch = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Batch Results (CSV)",
                    data=csv_batch,
                    file_name=f"batch_moderation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
