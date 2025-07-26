import streamlit as st
import time
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Configure page
st.set_page_config(
    page_title="Autonomous Research Orchestrator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Import our research system
try:
    from crew import ResearchOrchestrator
    from tools.chroma_memory import ChromaMemory
    SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"System import error: {e}")
    SYSTEM_AVAILABLE = False

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .research-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_research' not in st.session_state:
    st.session_state.current_research = None
if 'orchestrator' not in st.session_state and SYSTEM_AVAILABLE:
    try:
        st.session_state.orchestrator = ResearchOrchestrator()
        st.session_state.memory = ChromaMemory()
    except Exception as e:
        st.error(f"Failed to initialize research system: {e}")
        SYSTEM_AVAILABLE = False

def main():
    """Main application interface."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Autonomous Research Orchestrator</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Academic Research Assistant")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # System status
        st.subheader("System Status")
        if SYSTEM_AVAILABLE:
            st.markdown('<span class="status-success">🟢 System Online</span>', unsafe_allow_html=True)
            
            # Show memory stats if available
            if hasattr(st.session_state, 'orchestrator'):
                try:
                    memory_stats = st.session_state.orchestrator.get_memory_stats()
                    st.metric("Papers Stored", memory_stats.get('papers_stored', 0))
                    st.metric("Research Sessions", memory_stats.get('research_sessions', 0))
                except:
                    st.info("Memory stats unavailable")
        else:
            st.markdown('<span class="status-error">🔴 System Offline</span>', unsafe_allow_html=True)
            st.error("Please check your environment setup")
        
        # API Status
        st.subheader("API Status")
        groq_status = "🟢" if os.getenv('GROQ_API_KEY') else "🔴"
        gemini_status = "🟢" if os.getenv('GEMINI_API_KEY') else "🔴"
        openai_status = "🟢" if os.getenv('OPENAI_API_KEY') else "🔴"
        
        st.markdown(f"Groq API: {groq_status}")
        st.markdown(f"Gemini API: {gemini_status}")
        st.markdown(f"OpenAI API: {openai_status}")
        
        # Research History
        st.subheader("📚 Research History")
        if st.session_state.research_history:
            for i, research in enumerate(reversed(st.session_state.research_history[-5:])):
                with st.expander(f"Research {len(st.session_state.research_history)-i}"):
                    st.write(f"**Query:** {research['query'][:50]}...")
                    st.write(f"**Status:** {research['status']}")
                    st.write(f"**Time:** {research['timestamp']}")
        else:
            st.info("No research history yet")
    
    # Main content area
    if not SYSTEM_AVAILABLE:
        st.error("🚨 System not available. Please check your setup and try again.")
        return
    
    # Research Interface
    st.header("🔍 Research Interface")
    
    # Research input form
    with st.form("research_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            research_query = st.text_area(
                "Enter your research question:",
                placeholder="e.g., What are the latest developments in transformer architectures for computer vision?",
                height=100
            )
        
        with col2:
            research_mode = st.selectbox(
                "Research Mode:",
                ["Quick Research", "Comprehensive Research", "Literature Review"],
                help="Quick: Fast analysis (2-3 min)\nComprehensive: Deep analysis (5-10 min)\nLiterature Review: Academic review"
            )
            
            max_papers = st.slider("Max Papers:", 5, 25, 10)
            
            report_format = st.selectbox(
                "Report Format:",
                ["Brief", "Comprehensive", "Executive", "Technical"]
            )
        
        submitted = st.form_submit_button("🚀 Start Research", use_container_width=True)
    
    # Execute research
    if submitted and research_query.strip():
        execute_research(research_query, research_mode, max_papers, report_format)
    
    # Display current research if running
    if st.session_state.current_research:
        display_research_progress()
    
    # Research results display
    if st.session_state.research_history:
        st.header("📊 Latest Research Results")
        display_latest_results()

def execute_research(query, mode, max_papers, report_format):
    """Execute research based on user inputs."""
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Store current research info
    research_info = {
        'query': query,
        'mode': mode,
        'max_papers': max_papers,
        'report_format': report_format,
        'status': 'running',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'progress': 0
    }
    st.session_state.current_research = research_info
    
    try:
        # Step 1: Initialize
        status_text.text("🔄 Initializing research workflow...")
        progress_bar.progress(10)
        time.sleep(1)
        
        # Step 2: Planning
        status_text.text("🧠 Analyzing query and planning research strategy...")
        progress_bar.progress(20)
        time.sleep(1)
        
        # Step 3: Search
        status_text.text("🔍 Searching arXiv for relevant papers...")
        progress_bar.progress(40)
        
        if mode == "Quick Research":
            result = st.session_state.orchestrator.execute_quick_research(query, max_papers)
        elif mode == "Literature Review":
            result = st.session_state.orchestrator.create_literature_review(query)
        else:
            # Comprehensive research (simplified for demo)
            result = st.session_state.orchestrator.execute_quick_research(query, max_papers)
        
        progress_bar.progress(60)
        
        # Step 4: Analysis
        status_text.text("🔬 Analyzing papers and extracting insights...")
        progress_bar.progress(80)
        time.sleep(1)
        
        # Step 5: Report Generation
        status_text.text("📝 Generating research report...")
        progress_bar.progress(90)
        time.sleep(1)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Research completed successfully!")
        
        # Store results
        research_info.update({
            'status': 'completed',
            'result': result,
            'progress': 100
        })
        
        st.session_state.research_history.append(research_info)
        st.session_state.current_research = None
        
        # Success message
        st.success("🎉 Research completed successfully!")
        st.balloons()
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Research failed")
        
        research_info.update({
            'status': 'failed',
            'error': str(e),
            'progress': 0
        })
        
        st.session_state.research_history.append(research_info)
        st.session_state.current_research = None
        
        st.error(f"Research failed: {str(e)}")

def display_research_progress():
    """Display current research progress."""
    
    research = st.session_state.current_research
    
    st.subheader("🔄 Research in Progress")
    st.write(f"**Query:** {research['query']}")
    st.write(f"**Mode:** {research['mode']}")
    
    # Progress bar
    progress = research.get('progress', 0)
    st.progress(progress / 100)
    
    if progress < 100:
        st.info("Research is running... Please wait.")
    else:
        st.success("Research completed!")

def display_latest_results():
    """Display the latest research results."""
    
    latest_research = st.session_state.research_history[-1]
    
    if latest_research['status'] != 'completed':
        return
    
    result = latest_research['result']
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔍 Search Results", "🔬 Analysis", "📄 Full Report"])
    
    with tab1:
        st.subheader("Research Summary")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Research Mode", latest_research['mode'])
        with col2:
            st.metric("Papers Found", "10+")  # Would extract from actual results
        with col3:
            st.metric("Status", latest_research['status'].title())
        with col4:
            st.metric("Duration", "~2 min")  # Would calculate actual duration
        
        # Quick insights
        if result.get('summary'):
            st.markdown("### Key Insights")
            summary_text = result['summary']
            st.markdown(summary_text[:500] + "..." if len(summary_text) > 500 else summary_text)
    
    with tab2:
        st.subheader("Search Results")
        if result.get('search_results'):
            search_text = result['search_results']
            st.markdown(search_text[:2000] + "..." if len(search_text) > 2000 else search_text)
        else:
            st.info("No search results available")
    
    with tab3:
        st.subheader("Analysis Results")
        if result.get('analysis'):
            analysis_text = result['analysis']
            st.markdown(analysis_text[:2000] + "..." if len(analysis_text) > 2000 else analysis_text)
        else:
            st.info("No analysis results available")
    
    with tab4:
        st.subheader("Complete Research Report")
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Download as Markdown"):
                download_report(result, "markdown")
        with col2:
            if st.button("📑 Download as Text"):
                download_report(result, "text")
        
        # Display full report
        if result.get('summary'):
            st.markdown("---")
            st.markdown(result['summary'])

def download_report(result, format_type):
    """Handle report downloads."""
    
    if format_type == "markdown":
        report_content = f"""# Research Report
        
## Query
{st.session_state.research_history[-1]['query']}

## Summary
{result.get('summary', 'No summary available')}

## Search Results
{result.get('search_results', 'No search results available')}

## Analysis
{result.get('analysis', 'No analysis available')}

---
Generated by Autonomous Research Orchestrator
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    else:
        report_content = f"""Research Report
===============

Query: {st.session_state.research_history[-1]['query']}

Summary:
{result.get('summary', 'No summary available')}

Search Results:
{result.get('search_results', 'No search results available')}

Analysis:
{result.get('analysis', 'No analysis available')}

Generated by Autonomous Research Orchestrator
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    st.download_button(
        label=f"📥 Download {format_type.title()} Report",
        data=report_content,
        file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{'md' if format_type == 'markdown' else 'txt'}",
        mime="text/plain",
        use_container_width=True
    )

# Demo section
def show_demo_section():
    """Show demo section with sample queries."""
    
    st.header("🎯 Try These Sample Queries")
    
    sample_queries = [
        "What are the latest developments in transformer architectures for computer vision?",
        "How are large language models being applied to scientific research?",
        "What are the current trends in federated learning and privacy-preserving AI?",
        "What are the recent advances in graph neural networks for drug discovery?",
        "How is reinforcement learning being used in robotics applications?"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, query in enumerate(sample_queries):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            if st.button(f"🔬 {query[:50]}...", key=f"demo_{i}", use_container_width=True):
                st.session_state.demo_query = query
                st.experimental_rerun()

# Footer
def show_footer():
    """Show application footer."""
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🔬 <strong>Autonomous Research Orchestrator</strong> v1.0</p>
        <p>Built with CrewAI, Groq, Gemini, ChromaDB & Streamlit</p>
        <p>Empowering researchers with AI-driven insights 🚀</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Check for demo query
    if hasattr(st.session_state, 'demo_query'):
        st.text_area("Research Query", value=st.session_state.demo_query, key="demo_input")
        del st.session_state.demo_query
    
    # Run main app
    main()
    
    # Show demo section
    if not st.session_state.research_history:
        show_demo_section()
    
    # Show footer
    show_footer()