import streamlit as st
import json
from src.main import AutonomousResearchOrchestrator
from src.utils.config import Config

# Page config
st.set_page_config(
    page_title="Autonomous Research Orchestrator",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if "orchestrator" not in st.session_state:
    try:
        st.session_state.orchestrator = AutonomousResearchOrchestrator()
        st.session_state.ready = True
    except ValueError as e:
        st.session_state.ready = False
        st.session_state.error = str(e)

# Main interface
st.title("🔬 Autonomous Research Orchestrator")
st.markdown("Multi-Agent AI Research System")

# Check if system is ready
if not st.session_state.get("ready", False):
    st.error(f"System not ready: {st.session_state.get('error', 'Unknown error')}")
    st.info("Please check your .env file and ensure all API keys are set.")
    st.stop()

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.info(f"Groq Model: {Config.GROQ_MODEL}")
st.sidebar.info(f"Gemini Model: {Config.GEMINI_MODEL}")

# Main input
query = st.text_input("Enter your research query:", placeholder="artificial intelligence trends in healthcare")

if st.button("Start Research", type="primary"):
    if query:
        with st.spinner("Research in progress..."):
            # Execute research
            results = st.session_state.orchestrator.research(query)
            
            # Display results
            if results.get("status") == "completed":
                st.success("Research completed successfully!")
                
                # Show workflow
                st.subheader("Workflow Executed")
                workflow = results.get("workflow", [])
                st.write(" → ".join(workflow))
                
                # Show agent results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Agent Results")
                    for result in results.get("results", []):
                        agent_name = result.get("agent", "unknown")
                        with st.expander(f"{agent_name.title()} Agent"):
                            if agent_name == "research":
                                st.write("**Findings:**")
                                st.write(result.get("findings", "No findings"))
                            elif agent_name == "analysis":
                                st.write("**Analysis:**")
                                st.write(result.get("analysis", "No analysis"))
                            elif agent_name == "summary":
                                st.write("**Summary:**")
                                st.write(result.get("summary", "No summary"))
                            elif agent_name == "memory":
                                st.write("**Context Analysis:**")
                                st.write(result.get("context_analysis", "No context"))
                
                with col2:
                    st.subheader("Final Summary")
                    final_summary = st.session_state.orchestrator.get_summary(results)
                    st.write(final_summary)
                
                # Raw results (expandable)
                with st.expander("View Raw Results"):
                    st.json(results)
                    
            else:
                st.error(f"Research failed: {results.get('error', 'Unknown error')}")
    else:
        st.warning("Please enter a research query.")

# Footer
st.markdown("---")
st.markdown("Built with LangChain v0.3 • Groq • Gemini • ChromaDB")