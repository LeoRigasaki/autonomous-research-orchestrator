from crewai import Agent
from crewai.tools import tool
import os
import json
import re
from typing import List, Dict, Optional
from groq import Groq
from google.generativeai import configure, GenerativeModel


class AnalysisAgent:
    """Analysis Agent specialized in processing and analyzing academic papers."""
    
    def __init__(self):
        self.groq_client = None
        self.gemini_model = None
        self._setup_llms()
    
    def _setup_llms(self):
        """Initialize LLM clients with fallback support."""
        try:
            # Setup Groq as primary
            if os.getenv('GROQ_API_KEY'):
                self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                
            # Setup Gemini as fallback (better for long context)
            if os.getenv('GEMINI_API_KEY'):
                configure(api_key=os.getenv('GEMINI_API_KEY'))
                self.gemini_model = GenerativeModel('gemini-1.5-flash')
                
        except Exception as e:
            print(f"Warning: LLM setup error in AnalysisAgent: {e}")
    
    def create_agent(self, tools: list = None) -> Agent:
        """
        Create the Analysis Agent with specified tools.
        
        Args:
            tools: List of tools available to the agent
            
        Returns:
            Configured CrewAI Agent
        """
        if tools is None:
            tools = [self.create_paper_analysis_tool()]
        
        return Agent(
            role="Academic Paper Analysis Expert",
            goal="Analyze academic papers to extract key insights, methodologies, and research contributions",
            backstory="""You are a seasoned academic researcher and analyst with expertise across 
            multiple domains including AI, machine learning, computer science, and related fields. 
            You have published numerous papers and have experience reviewing for top-tier conferences.
            
            Your expertise includes:
            - Understanding research methodologies and experimental designs
            - Identifying key contributions and novelty in academic work
            - Analyzing technical approaches and their limitations
            - Extracting actionable insights from complex research papers
            - Comparing and contrasting different research approaches
            - Identifying research gaps and future directions
            
            You excel at breaking down complex technical content into clear, actionable insights 
            and can identify the most important aspects of research that would be valuable for 
            practitioners and other researchers.""",
            verbose=True,
            allow_delegation=False,
            tools=tools,
            max_iter=3,
            memory=True,
            llm="gemini/gemini-1.5-flash"  # Use Gemini for long context analysis
        )
    
    def create_paper_analysis_tool(self):
        """Create paper analysis tool for the agent."""
        
        @tool("analyze_papers")
        def paper_analysis_tool(papers_text: str, analysis_focus: str = "general") -> str:
            """
            Analyze academic papers and extract key insights.
            
            Args:
                papers_text: Text containing paper information (titles, abstracts, etc.)
                analysis_focus: Focus area (general, methodology, results, trends)
                
            Returns:
                Comprehensive analysis of the papers
            """
            return self.analyze_papers_content(papers_text, analysis_focus)
        
        return paper_analysis_tool
    
    def analyze_papers_content(self, papers_text: str, focus: str = "general") -> str:
        """
        Analyze paper content using LLMs.
        
        Args:
            papers_text: Text containing paper information
            focus: Analysis focus area
            
        Returns:
            Detailed analysis
        """
        analysis_prompt = f"""
        Analyze the following academic papers and provide a comprehensive analysis focused on: {focus}
        
        Papers to analyze:
        {papers_text}
        
        Please provide:
        
        1. **Key Research Themes**: What are the main research directions represented?
        2. **Methodological Approaches**: What methods and techniques are being used?
        3. **Novel Contributions**: What new ideas, algorithms, or insights are presented?
        4. **Research Gaps**: What areas need further investigation?
        5. **Practical Applications**: How can this research be applied in practice?
        6. **Future Directions**: What are the promising next steps?
        7. **Quality Assessment**: Evaluate the overall quality and impact potential
        
        Format the response with clear sections and actionable insights.
        """
        
        try:
            # Try Gemini first for long context analysis
            if self.gemini_model and len(papers_text) > 2000:
                response = self.gemini_model.generate_content(analysis_prompt)
                return response.text
            
            # Fallback to Groq for shorter content
            elif self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            else:
                return "LLM not available for analysis"
                
        except Exception as e:
            return f"Error during analysis: {str(e)}"
    
    def extract_paper_insights(self, papers: List[Dict]) -> Dict:
        """
        Extract structured insights from a list of papers.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Structured insights dictionary
        """
        if not papers:
            return {"error": "No papers provided"}
        
        # Create summary text for analysis
        papers_text = self._format_papers_for_analysis(papers)
        
        # Perform different types of analysis
        insights = {
            "general_analysis": self.analyze_papers_content(papers_text, "general"),
            "methodology_analysis": self.analyze_papers_content(papers_text, "methodology"),
            "trends_analysis": self.analyze_papers_content(papers_text, "trends"),
            "paper_count": len(papers),
            "categories": self._extract_categories(papers),
            "authors": self._extract_top_authors(papers),
            "publication_timeline": self._analyze_publication_timeline(papers)
        }
        
        return insights
    
    def _format_papers_for_analysis(self, papers: List[Dict]) -> str:
        """Format papers into text for LLM analysis."""
        formatted_text = ""
        
        for i, paper in enumerate(papers[:10], 1):  # Limit to top 10 for analysis
            formatted_text += f"\n## Paper {i}\n"
            formatted_text += f"**Title:** {paper.get('title', 'N/A')}\n"
            formatted_text += f"**Authors:** {', '.join(paper.get('authors', [])[:3])}\n"
            formatted_text += f"**Published:** {paper.get('published', 'N/A')[:10]}\n"
            formatted_text += f"**Categories:** {', '.join(paper.get('categories', []))}\n"
            formatted_text += f"**Abstract:** {paper.get('summary', 'N/A')[:500]}...\n"
            formatted_text += "---\n"
        
        return formatted_text
    
    def _extract_categories(self, papers: List[Dict]) -> Dict:
        """Extract and count paper categories."""
        categories = {}
        for paper in papers:
            for cat in paper.get('categories', []):
                categories[cat] = categories.get(cat, 0) + 1
        
        # Sort by frequency
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_categories[:10])  # Top 10 categories
    
    def _extract_top_authors(self, papers: List[Dict]) -> Dict:
        """Extract and count top authors."""
        authors = {}
        for paper in papers:
            for author in paper.get('authors', []):
                authors[author] = authors.get(author, 0) + 1
        
        # Sort by frequency
        sorted_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_authors[:10])  # Top 10 authors
    
    def _analyze_publication_timeline(self, papers: List[Dict]) -> Dict:
        """Analyze publication timeline."""
        years = {}
        for paper in papers:
            try:
                pub_date = paper.get('published', '')
                if pub_date:
                    year = pub_date[:4]
                    years[year] = years.get(year, 0) + 1
            except:
                continue
        
        sorted_years = sorted(years.items())
        return dict(sorted_years)
    
    def compare_papers(self, papers: List[Dict], comparison_aspects: List[str] = None) -> str:
        """
        Compare multiple papers across different aspects.
        
        Args:
            papers: List of papers to compare
            comparison_aspects: Specific aspects to compare
            
        Returns:
            Comparison analysis
        """
        if len(papers) < 2:
            return "Need at least 2 papers for comparison"
        
        if comparison_aspects is None:
            comparison_aspects = ["methodology", "results", "novelty", "limitations"]
        
        papers_text = self._format_papers_for_analysis(papers)
        
        comparison_prompt = f"""
        Compare the following academic papers across these aspects: {', '.join(comparison_aspects)}
        
        Papers:
        {papers_text}
        
        Provide a detailed comparison that includes:
        1. Similarities and differences in approach
        2. Strengths and weaknesses of each paper
        3. Which paper makes the strongest contribution and why
        4. How the papers complement or contradict each other
        5. Synthesis of insights from all papers
        
        Format as a structured comparison with clear sections.
        """
        
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(comparison_prompt)
                return response.text
            elif self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": comparison_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            else:
                return "LLM not available for comparison"
                
        except Exception as e:
            return f"Error during comparison: {str(e)}"
    
    def identify_research_gaps(self, papers: List[Dict]) -> str:
        """
        Identify research gaps based on analyzed papers.
        
        Args:
            papers: List of papers to analyze
            
        Returns:
            Research gaps analysis
        """
        papers_text = self._format_papers_for_analysis(papers)
        
        gaps_prompt = f"""
        Based on these academic papers, identify key research gaps and opportunities:
        
        Papers:
        {papers_text}
        
        Please identify:
        1. **Methodological Gaps**: What approaches haven't been tried?
        2. **Application Gaps**: What domains or use cases are underexplored?
        3. **Technical Limitations**: What current limitations need addressing?
        4. **Evaluation Gaps**: What aspects need better evaluation methods?
        5. **Future Opportunities**: What are the most promising research directions?
        
        Prioritize gaps by potential impact and feasibility.
        """
        
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(gaps_prompt)
                return response.text
            elif self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": gaps_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.4,  # Slightly more creative for gap identification
                    max_tokens=1500
                )
                return response.choices[0].message.content
            else:
                return "LLM not available for gap analysis"
                
        except Exception as e:
            return f"Error during gap analysis: {str(e)}"


def create_analysis_agent_tool():
    """Create a tool wrapper for Analysis Agent functionality."""
    
    analysis_agent = AnalysisAgent()
    
    @tool("comprehensive_paper_analysis")
    def comprehensive_analysis_tool(papers_text: str, analysis_type: str = "general") -> str:
        """
        Perform comprehensive analysis of academic papers.
        
        Args:
            papers_text: Text containing paper information (formatted from search results)
            analysis_type: Type of analysis (general, methodology, trends, gaps, comparison)
            
        Returns:
            Detailed analysis results
        """
        if analysis_type == "gaps":
            # Parse papers from text (simplified parsing)
            papers = analysis_agent._parse_papers_from_text(papers_text)
            return analysis_agent.identify_research_gaps(papers)
        elif analysis_type == "comparison":
            papers = analysis_agent._parse_papers_from_text(papers_text)
            return analysis_agent.compare_papers(papers)
        else:
            return analysis_agent.analyze_papers_content(papers_text, analysis_type)
    
    return comprehensive_analysis_tool


# Add helper method to AnalysisAgent class
def _parse_papers_from_text(self, papers_text: str) -> List[Dict]:
    """
    Parse papers from formatted text (basic implementation).
    
    Args:
        papers_text: Formatted text containing paper information
        
    Returns:
        List of paper dictionaries
    """
    papers = []
    
    # Split by paper sections (looking for numbered papers)
    paper_sections = re.split(r'#{1,2}\s*\d+\.', papers_text)
    
    for section in paper_sections[1:]:  # Skip first empty section
        paper = {}
        
        # Extract title
        title_match = re.search(r'\*\*Title:\*\*\s*(.+)', section)
        if title_match:
            paper['title'] = title_match.group(1).strip()
        
        # Extract authors
        authors_match = re.search(r'\*\*Authors:\*\*\s*(.+)', section)
        if authors_match:
            paper['authors'] = [a.strip() for a in authors_match.group(1).split(',')]
        
        # Extract categories
        cats_match = re.search(r'\*\*Categories:\*\*\s*(.+)', section)
        if cats_match:
            paper['categories'] = [c.strip() for c in cats_match.group(1).split(',')]
        
        # Extract published date
        pub_match = re.search(r'\*\*Published:\*\*\s*(.+)', section)
        if pub_match:
            paper['published'] = pub_match.group(1).strip()
        
        # Extract abstract
        abstract_match = re.search(r'\*\*Abstract:\*\*\s*(.+?)(?:\n|$)', section, re.DOTALL)
        if abstract_match:
            paper['summary'] = abstract_match.group(1).strip()
        
        if paper.get('title'):  # Only add if we found a title
            papers.append(paper)
    
    return papers

# Add the helper method to the AnalysisAgent class
AnalysisAgent._parse_papers_from_text = _parse_papers_from_text