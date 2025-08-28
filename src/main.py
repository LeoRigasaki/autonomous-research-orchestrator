from src.agents.supervisor import SupervisorAgent
from src.utils.config import Config
import json
from typing import Dict, Any

class AutonomousResearchOrchestrator:
    def __init__(self):
        print("DEBUG: Starting orchestrator initialization...")
        
        # Validate API keys
        print("DEBUG: Validating API keys...")
        Config.validate_keys()
        print("DEBUG: API keys validated")
        
        # Initialize supervisor
        print("DEBUG: Initializing supervisor...")
        self.supervisor = SupervisorAgent()
        print("DEBUG: Supervisor initialized successfully")
    
    def research(self, query: str) -> Dict[str, Any]:
        """Main research function"""
        try:
            print(f"DEBUG: Starting research for: {query}")
            results = self.supervisor.execute_workflow(query)
            print("DEBUG: Research completed successfully")
            return results
        except Exception as e:
            print(f"DEBUG: Research failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "status": "failed",
                "query": query
            }
    
    def get_summary(self, results: Dict[str, Any]) -> str:
        """Extract final summary from results"""
        if results.get("status") == "failed":
            return f"Research failed: {results.get('error')}"
        
        # Find summary result
        for result in results.get("results", []):
            if result.get("agent") == "summary":
                return result.get("summary", "No summary generated")
        
        return "Summary not available"

def main():
    # Initialize orchestrator
    orchestrator = AutonomousResearchOrchestrator()
    
    # Test query
    query = "mcp server"
    
    print(f"Research Query: {query}")
    print("-" * 50)
    
    # Execute research
    results = orchestrator.research(query)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Print summary
    print("\nFinal Summary:")
    print("-" * 30)
    print(orchestrator.get_summary(results))

if __name__ == "__main__":
    main()