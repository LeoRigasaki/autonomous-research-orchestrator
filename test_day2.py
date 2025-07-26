#!/usr/bin/env python3
"""
Day 2 Test Script - Autonomous Research Orchestrator
Test the core agent framework and arXiv integration.
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
    
    print(f"GROQ_API_KEY: {'✅ Found' if groq_key else '❌ Missing'}")
    print(f"GEMINI_API_KEY: {'✅ Found' if gemini_key else '❌ Missing'}")
    
    if not groq_key and not gemini_key:
        print("⚠️  Warning: No API keys found. Some features may not work.")
    
    return groq_key or gemini_key


def test_arxiv_api():
    """Test arXiv API integration."""
    print("\n📚 Testing arXiv API")
    print("-" * 40)
    
    try:
        from tools.arxiv_api import ArxivAPI
        
        # Create client
        arxiv_client = ArxivAPI(max_results=3)
        print("✅ ArxivAPI client created")
        
        # Test search
        print("🔍 Testing paper search...")
        papers = arxiv_client.search_papers("machine learning", max_results=2)
        
        if papers:
            print(f"✅ Found {len(papers)} papers")
            print(f"   First paper: {papers[0]['title'][:50]}...")
            return True
        else:
            print("❌ No papers found")
            return False
            
    except Exception as e:
        print(f"❌ arXiv API test failed: {str(e)}")
        return False


def test_lead_agent():
    """Test Lead Agent functionality."""
    print("\n🧠 Testing Lead Agent")
    print("-" * 40)
    
    try:
        from agents.lead_agent import LeadAgent
        
        # Create lead agent
        lead_agent = LeadAgent()
        print("✅ Lead Agent created")
        
        # Test query analysis
        test_query = "deep learning for computer vision"
        print(f"🔍 Analyzing query: '{test_query}'")
        
        analysis = lead_agent.analyze_query(test_query)
        if analysis and 'analysis' in analysis:
            print("✅ Query analysis completed")
            print(f"   Source: {analysis.get('source', 'unknown')}")
            return True
        else:
            print("❌ Query analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Lead Agent test failed: {str(e)}")
        return False


def test_search_agent():
    """Test Search Agent functionality."""
    print("\n🔎 Testing Search Agent")
    print("-" * 40)
    
    try:
        from agents.search_agent import SearchAgent
        
        # Create search agent
        search_agent = SearchAgent()
        print("✅ Search Agent created")
        
        # Test search execution
        test_query = "neural networks"
        print(f"🔍 Executing search for: '{test_query}'")
        
        results = search_agent.execute_comprehensive_search(test_query)
        if results and len(results) > 100:  # Should be substantial text
            print("✅ Comprehensive search completed")
            print(f"   Result length: {len(results)} characters")
            return True
        else:
            print("❌ Search execution failed or returned minimal results")
            return False
            
    except Exception as e:
        print(f"❌ Search Agent test failed: {str(e)}")
        return False


def test_crew_integration():
    """Test crew integration and workflow."""
    print("\n🤝 Testing Crew Integration")
    print("-" * 40)
    
    try:
        from crew import create_research_crew
        
        # Create research crew
        orchestrator = create_research_crew()
        print("✅ Research crew created")
        
        # Test quick search
        test_query = "transformers in NLP"
        print(f"🔍 Testing quick search: '{test_query}'")
        
        result = orchestrator.quick_search(test_query, max_results=2)
        if result['status'] == 'success':
            print("✅ Quick search successful")
            print(f"   Result type: {result.get('type', 'unknown')}")
            return True
        else:
            print(f"❌ Quick search failed: {result.get('error', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Crew integration test failed: {str(e)}")
        return False


def run_full_test_suite():
    """Run the complete test suite."""
    print("🚀 Day 2 Test Suite - Autonomous Research Orchestrator")
    print("=" * 60)
    
    tests = [
        ("Environment Setup", test_environment),
        ("arXiv API", test_arxiv_api),
        ("Lead Agent", test_lead_agent),
        ("Search Agent", test_search_agent),
        ("Crew Integration", test_crew_integration)
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
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Day 2 implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = run_full_test_suite()
    sys.exit(0 if success else 1)