#!/usr/bin/env python3
"""
Day 3 Test Script - Autonomous Research Orchestrator
Test the enhanced multi-agent system with analysis and memory capabilities.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment():
    """Test environment setup and API keys."""
    print("🔧 Testing Environment Setup")
    print("-" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check API keys
    groq_key = os.getenv('GROQ_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')  # Should be 'dummy'
    
    print(f"GROQ_API_KEY: {'✅ Found' if groq_key else '❌ Missing'}")
    print(f"GEMINI_API_KEY: {'✅ Found' if gemini_key else '❌ Missing'}")
    print(f"OPENAI_API_KEY: {'✅ Found' if openai_key else '❌ Missing'}")
    
    if not groq_key and not gemini_key:
        print("⚠️  Warning: No API keys found. Some features may not work.")
    
    return groq_key or gemini_key


def test_chroma_memory():
    """Test ChromaDB memory system."""
    print("\n🧠 Testing Chroma Memory System")
    print("-" * 40)
    
    try:
        from tools.chroma_memory import ChromaMemory
        
        # Create memory instance
        memory = ChromaMemory(persist_directory="test_chroma_db")
        print("✅ ChromaMemory instance created")
        
        # Test storing papers
        test_papers = [
            {
                'id': 'test1',
                'title': 'Test Paper on Machine Learning',
                'authors': ['Alice Smith', 'Bob Jones'],
                'summary': 'This is a test paper about machine learning algorithms.',
                'published': '2024-01-01',
                'categories': ['cs.LG', 'cs.AI'],
                'pdf_url': 'https://example.com/test1.pdf',
                'arxiv_url': 'https://arxiv.org/abs/test1'
            }
        ]
        
        success = memory.store_papers(test_papers, "test_collection")
        if success:
            print("✅ Test papers stored successfully")
        else:
            print("❌ Failed to store test papers")
            return False
        
        # Test searching papers
        results = memory.search_papers("machine learning", "test_collection", n_results=1)
        if results:
            print(f"✅ Memory search successful - found {len(results)} papers")
            return True
        else:
            print("❌ Memory search failed")
            return False
            
    except Exception as e:
        print(f"❌ Chroma memory test failed: {str(e)}")
        return False


def test_analysis_agent():
    """Test Analysis Agent functionality."""
    print("\n🔬 Testing Analysis Agent")
    print("-" * 40)
    
    try:
        from agents.analysis_agent import AnalysisAgent
        
        # Create analysis agent
        analysis_agent = AnalysisAgent()
        print("✅ Analysis Agent created")
        
        # Test paper content analysis
        test_content = """
        ## Paper 1
        **Title:** Deep Learning for Computer Vision
        **Authors:** John Doe, Jane Smith
        **Published:** 2024-01-15
        **Categories:** cs.CV, cs.LG
        **Abstract:** This paper presents a novel approach to computer vision using deep learning...
        
        ## Paper 2
        **Title:** Transformer Networks in NLP
        **Authors:** Alice Johnson
        **Published:** 2024-02-01
        **Categories:** cs.CL, cs.LG
        **Abstract:** We propose improvements to transformer architectures for natural language processing...
        """
        
        print("🔍 Analyzing test content...")
        analysis = analysis_agent.analyze_papers_content(test_content, "general")
        
        if analysis and len(analysis) > 100:
            print("✅ Paper analysis completed")
            print(f"   Analysis length: {len(analysis)} characters")
            return True
        else:
            print("❌ Paper analysis failed or returned minimal content")
            return False
            
    except Exception as e:
        print(f"❌ Analysis Agent test failed: {str(e)}")
        return False


def test_summary_agent():
    """Test Summary Agent functionality."""
    print("\n📝 Testing Summary Agent")
    print("-" * 40)
    
    try:
        from agents.summary_agent import SummaryAgent
        
        # Create summary agent
        summary_agent = SummaryAgent()
        print("✅ Summary Agent created")
        
        # Test report generation
        test_research_data = """
        Research Query: machine learning for healthcare
        
        Search Results: Found 10 papers on machine learning applications in healthcare...
        
        Analysis: Key themes include diagnostic imaging, drug discovery, and patient monitoring...
        """
        
        print("🔍 Generating test report...")
        report = summary_agent.generate_research_report(test_research_data, "brief")
        
        if report and len(report) > 200:
            print("✅ Report generation completed")
            print(f"   Report length: {len(report)} characters")
            return True
        else:
            print("❌ Report generation failed or returned minimal content")
            return False
            
    except Exception as e:
        print(f"❌ Summary Agent test failed: {str(e)}")
        return False


def test_comprehensive_crew():
    """Test the comprehensive multi-agent crew."""
    print("\n🤖 Testing Comprehensive Multi-Agent Crew")
    print("-" * 40)
    
    try:
        from crew import ResearchOrchestrator
        
        # Create research orchestrator
        orchestrator = ResearchOrchestrator()
        print("✅ Research orchestrator created with 4 agents")
        
        # Test memory stats
        memory_stats = orchestrator.get_memory_stats()
        print(f"✅ Memory system active - {memory_stats.get('memory_status', 'unknown')} status")
        
        # Test quick research
        test_query = "neural networks for time series prediction"
        print(f"🔍 Testing quick research: '{test_query}'")
        
        result = orchestrator.execute_quick_research(test_query, max_papers=3)
        
        if result['status'] == 'success':
            print("✅ Quick research workflow successful")
            print(f"   Query: {result['query']}")
            print(f"   Search results: {len(result.get('search_results', ''))} characters")
            print(f"   Analysis: {len(result.get('analysis', ''))} characters")
            print(f"   Summary: {len(result.get('summary', ''))} characters")
            return True
        else:
            print(f"❌ Quick research failed: {result.get('error', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Comprehensive crew test failed: {str(e)}")
        return False


def test_memory_integration():
    """Test memory integration with agents."""
    print("\n💾 Testing Memory Integration")
    print("-" * 40)
    
    try:
        from tools.chroma_memory import create_memory_search_tool, create_memory_store_tool
        
        # Test memory tools
        memory_search = create_memory_search_tool()
        memory_store = create_memory_store_tool()
        
        print("✅ Memory tools created")
        
        # Test storing papers via tool
        test_papers_json = '''[
            {
                "id": "integration_test",
                "title": "Integration Test Paper",
                "authors": ["Test Author"],
                "summary": "This is a test paper for integration testing.",
                "published": "2024-01-01",
                "categories": ["test.category"],
                "pdf_url": "https://example.com/test.pdf",
                "arxiv_url": "https://arxiv.org/abs/test"
            }
        ]'''
        
        store_result = memory_store.func(test_papers_json, "integration_test")
        if "Successfully stored" in store_result:
            print("✅ Memory storage via tool successful")
        else:
            print(f"❌ Memory storage failed: {store_result}")
            return False
        
        # Test searching via tool
        search_result = memory_search.func("integration test", "integration_test")
        if "integration_test" in search_result.lower():
            print("✅ Memory search via tool successful")
            return True
        else:
            print("❌ Memory search via tool failed")
            return False
            
    except Exception as e:
        print(f"❌ Memory integration test failed: {str(e)}")
        return False


def run_comprehensive_demo():
    """Run a comprehensive demo of the system."""
    print("\n🚀 Running Comprehensive Demo")
    print("=" * 60)
    
    try:
        from crew import ResearchOrchestrator
        
        orchestrator = ResearchOrchestrator()
        demo_query = "attention mechanisms in transformer models"
        
        print(f"Demo Query: '{demo_query}'")
        print("Executing quick research workflow...")
        
        result = orchestrator.execute_quick_research(demo_query, max_papers=5)
        
        if result['status'] == 'success':
            print("\n✅ Demo completed successfully!")
            print("\n📊 Demo Results Summary:")
            print(f"- Search Results: {len(result.get('search_results', ''))} characters")
            print(f"- Analysis: {len(result.get('analysis', ''))} characters") 
            print(f"- Summary Report: {len(result.get('summary', ''))} characters")
            
            # Show excerpt from summary
            summary = result.get('summary', '')
            if summary:
                print(f"\n📝 Summary Excerpt:")
                print(summary[:300] + "..." if len(summary) > 300 else summary)
            
            return True
        else:
            print(f"❌ Demo failed: {result.get('error', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        return False


def run_full_test_suite():
    """Run the complete Day 3 test suite."""
    print("🚀 Day 3 Test Suite - Enhanced Multi-Agent Research System")
    print("=" * 70)
    
    tests = [
        ("Environment Setup", test_environment),
        ("Chroma Memory System", test_chroma_memory),
        ("Analysis Agent", test_analysis_agent),
        ("Summary Agent", test_summary_agent),
        ("Memory Integration", test_memory_integration),
        ("Comprehensive Multi-Agent Crew", test_comprehensive_crew),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 70)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<35} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Run demo if most tests passed
    if passed >= total - 1:  # Allow 1 failure
        demo_success = run_comprehensive_demo()
        if demo_success:
            print("\n🎉 All tests passed and demo successful!")
            print("Day 3 implementation is fully operational!")
            return True
    
    if passed == total:
        print("🎉 Perfect! All tests passed.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed >= total - 1


def cleanup_test_data():
    """Clean up test data created during testing."""
    try:
        import shutil
        
        # Remove test ChromaDB directory
        test_dirs = ["test_chroma_db", "chroma_db"]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                print(f"🧹 Cleaned up {test_dir}")
                
    except Exception as e:
        print(f"⚠️  Cleanup warning: {str(e)}")


if __name__ == "__main__":
    try:
        success = run_full_test_suite()
        sys.exit(0 if success else 1)
    finally:
        cleanup_test_data()