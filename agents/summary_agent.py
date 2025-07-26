from crewai import Agent
from crewai.tools import tool
import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from groq import Groq
from google.generativeai import configure, GenerativeModel


class SummaryAgent:
    """Summary Agent specialized in generating comprehensive research reports."""
    
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
                
            # Setup Gemini as fallback
            if os.getenv('GEMINI_API_KEY'):
                configure(api_key=os.getenv('GEMINI_API_KEY'))
                self.gemini_model = GenerativeModel('gemini-1.5-flash')
                
        except Exception as e:
            print(f"Warning: LLM setup error in SummaryAgent: {e}")
    
    def create_agent(self, tools: list = None) -> Agent:
        """
        Create the Summary Agent with specified tools.
        
        Args:
            tools: List of tools available to the agent
            
        Returns:
            Configured CrewAI Agent
        """
        if tools is None:
            tools = [self.create_report_generation_tool()]
        
        return Agent(
            role="Research Report Synthesis Expert",
            goal="Generate comprehensive, well-structured research reports that synthesize findings into actionable insights",
            backstory="""You are an expert research writer and synthesizer with extensive experience 
            in creating high-quality academic and industry research reports. You have worked with 
            leading research institutions and companies to transform complex research findings into 
            clear, actionable documents.
            
            Your expertise includes:
            - Synthesizing complex information from multiple sources
            - Creating clear, structured research reports
            - Identifying key insights and actionable recommendations
            - Writing for different audiences (academic, industry, executive)
            - Organizing information in logical, accessible formats
            - Highlighting the most important findings and implications
            
            You excel at taking disparate research findings and weaving them into coherent narratives 
            that help readers understand not just what was found, but what it means and how they can 
            act on it. Your reports are known for their clarity, structure, and practical value.""",
            verbose=True,
            allow_delegation=False,
            tools=tools,
            max_iter=3,
            memory=True,
            llm="groq/llama-3.3-70b-versatile"  # Good for structured output
        )
    
    def create_report_generation_tool(self):
        """Create report generation tool for the agent."""
        
        @tool("generate_research_report")
        def report_generation_tool(research_data: str, report_type: str = "comprehensive") -> str:
            """
            Generate a structured research report from analysis data.
            
            Args:
                research_data: Combined data from search and analysis phases
                report_type: Type of report (executive, comprehensive, technical, brief)
                
            Returns:
                Formatted research report
            """
            return self.generate_research_report(research_data, report_type)
        
        return report_generation_tool
    
    def generate_research_report(self, research_data: str, report_type: str = "comprehensive") -> str:
        """
        Generate a comprehensive research report.
        
        Args:
            research_data: Combined research findings and analysis
            report_type: Type of report to generate
            
        Returns:
            Formatted research report
        """
        report_prompts = {
            "executive": self._get_executive_report_prompt(),
            "comprehensive": self._get_comprehensive_report_prompt(),
            "technical": self._get_technical_report_prompt(),
            "brief": self._get_brief_report_prompt()
        }
        
        prompt_template = report_prompts.get(report_type, report_prompts["comprehensive"])
        
        full_prompt = f"""
        {prompt_template}
        
        Research Data to Synthesize:
        {research_data}
        
        Generate a well-structured report that synthesizes this information into actionable insights.
        Ensure the report is professional, clear, and valuable for readers.
        """
        
        try:
            # Use Groq for structured output
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": full_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=3000
                )
                return response.choices[0].message.content
            
            # Fallback to Gemini
            elif self.gemini_model:
                response = self.gemini_model.generate_content(full_prompt)
                return response.text
            
            else:
                return "LLM not available for report generation"
                
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def _get_executive_report_prompt(self) -> str:
        """Get prompt for executive summary report."""
        return """
        Create an EXECUTIVE SUMMARY research report with the following structure:
        
        # Executive Summary
        
        ## Key Findings (3-5 bullet points)
        - Most important discoveries
        
        ## Strategic Implications 
        - What this means for business/research strategy
        
        ## Recommended Actions
        - Specific, actionable next steps
        
        ## Research Overview
        - Brief methodology and scope summary
        
        Keep it concise, strategic, and action-oriented. Maximum 2 pages.
        """
    
    def _get_comprehensive_report_prompt(self) -> str:
        """Get prompt for comprehensive research report."""
        return """
        Create a COMPREHENSIVE research report with the following structure:
        
        # Research Report: [Topic]
        
        ## Executive Summary
        - Key findings overview
        - Main implications
        
        ## Research Methodology
        - Search strategy used
        - Papers analyzed
        - Analysis approach
        
        ## Key Findings
        ### Theme 1: [Major Theme]
        - Detailed findings
        - Supporting evidence
        
        ### Theme 2: [Major Theme]
        - Detailed findings
        - Supporting evidence
        
        ## Analysis and Insights
        - Cross-cutting themes
        - Novel contributions identified
        - Methodological trends
        
        ## Research Gaps and Opportunities
        - Identified gaps
        - Future research directions
        - Potential impact areas
        
        ## Practical Applications
        - How findings can be applied
        - Industry implications
        - Implementation considerations
        
        ## Conclusions and Recommendations
        - Summary of key insights
        - Specific recommendations
        - Next steps
        
        ## References and Further Reading
        - Key papers identified
        - Recommended follow-up resources
        
        Make it detailed, well-structured, and professionally formatted.
        """
    
    def _get_technical_report_prompt(self) -> str:
        """Get prompt for technical research report."""
        return """
        Create a TECHNICAL research report with the following structure:
        
        # Technical Research Analysis: [Topic]
        
        ## Abstract
        - Technical summary of findings
        
        ## Methodology Analysis
        - Technical approaches identified
        - Algorithms and methods used
        - Experimental designs
        
        ## Technical Findings
        - Detailed technical results
        - Performance comparisons
        - Implementation details
        
        ## Method Comparison
        - Comparative analysis of approaches
        - Strengths and limitations
        - Performance metrics
        
        ## Technical Gaps
        - Methodological limitations
        - Technical challenges identified
        - Areas needing development
        
        ## Implementation Recommendations
        - Technical best practices
        - Implementation guidelines
        - Tool and framework suggestions
        
        Focus on technical depth, methodological rigor, and implementation details.
        """
    
    def _get_brief_report_prompt(self) -> str:
        """Get prompt for brief research summary."""
        return """
        Create a BRIEF research summary with the following structure:
        
        # Research Brief: [Topic]
        
        ## What We Found
        - 3-5 key discoveries
        
        ## Why It Matters
        - Significance and implications
        
        ## What's Next
        - Recommended actions
        - Future directions
        
        ## Key Resources
        - Most important papers
        - Essential reading
        
        Keep it concise and accessible. Maximum 1 page.
        """
    
    def create_literature_review(self, papers_data: str, focus_area: str = "") -> str:
        """
        Create a literature review from research papers.
        
        Args:
            papers_data: Data about research papers
            focus_area: Specific area to focus the review on
            
        Returns:
            Formatted literature review
        """
        review_prompt = f"""
        Create a literature review focusing on: {focus_area if focus_area else "the research area"}
        
        Papers data:
        {papers_data}
        
        Structure the literature review as follows:
        
        # Literature Review: {focus_area if focus_area else "[Research Area]"}
        
        ## Introduction
        - Overview of the research area
        - Scope and objectives of this review
        
        ## Current State of Research
        - Major themes and approaches
        - Key methodologies
        - Important findings
        
        ## Critical Analysis
        - Strengths and limitations of current research
        - Methodological issues
        - Conflicting findings or debates
        
        ## Research Trends
        - Evolution of the field
        - Emerging directions
        - Technological advances
        
        ## Gaps and Future Directions
        - Identified research gaps
        - Promising future directions
        - Methodological improvements needed
        
        ## Conclusion
        - Summary of current state
        - Key takeaways
        - Implications for future work
        
        Make it scholarly, critical, and comprehensive.
        """
        
        try:
            if self.gemini_model:  # Use Gemini for longer content
                response = self.gemini_model.generate_content(review_prompt)
                return response.text
            elif self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": review_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=3000
                )
                return response.choices[0].message.content
            else:
                return "LLM not available for literature review"
                
        except Exception as e:
            return f"Error creating literature review: {str(e)}"
    
    def create_research_proposal(self, gap_analysis: str, topic: str) -> str:
        """
        Create a research proposal based on identified gaps.
        
        Args:
            gap_analysis: Analysis of research gaps
            topic: Research topic area
            
        Returns:
            Research proposal
        """
        proposal_prompt = f"""
        Based on the following gap analysis, create a research proposal for: {topic}
        
        Gap Analysis:
        {gap_analysis}
        
        Structure the research proposal as follows:
        
        # Research Proposal: [Specific Title]
        
        ## Abstract
        - Brief overview of proposed research
        
        ## Problem Statement
        - Current limitations and gaps
        - Why this research is needed
        
        ## Research Objectives
        - Primary objectives
        - Secondary objectives
        - Research questions
        
        ## Literature Background
        - Current state of research
        - Identified gaps
        
        ## Proposed Methodology
        - Research approach
        - Methods and techniques
        - Data requirements
        
        ## Expected Contributions
        - Novel contributions
        - Potential impact
        - Benefits to field
        
        ## Timeline and Milestones
        - Key phases
        - Deliverables
        - Success metrics
        
        ## Resource Requirements
        - Computational needs
        - Data requirements
        - Collaboration needs
        
        Make it compelling, feasible, and well-justified.
        """
        
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(proposal_prompt)
                return response.text
            elif self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": proposal_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.4,  # Slightly more creative for proposals
                    max_tokens=2500
                )
                return response.choices[0].message.content
            else:
                return "LLM not available for research proposal"
                
        except Exception as e:
            return f"Error creating research proposal: {str(e)}"
    
    def format_final_report(self, query: str, search_results: str, analysis: str) -> str:
        """
        Format the final research report combining all components.
        
        Args:
            query: Original research query
            search_results: Results from paper search
            analysis: Analysis results
            
        Returns:
            Complete formatted report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report_header = f"""
# Autonomous Research Report

**Research Query:** {query}  
**Generated:** {timestamp}  
**System:** Autonomous Research Orchestrator v1.0

---
        """
        
        combined_data = f"""
ORIGINAL QUERY: {query}

SEARCH RESULTS:
{search_results}

ANALYSIS RESULTS:
{analysis}
        """
        
        main_report = self.generate_research_report(combined_data, "comprehensive")
        
        return report_header + main_report


def create_summary_agent_tool():
    """Create a tool wrapper for Summary Agent functionality."""
    
    summary_agent = SummaryAgent()
    
    @tool("generate_final_report")
    def final_report_tool(query: str, search_results: str, analysis: str, 
                         report_type: str = "comprehensive") -> str:
        """
        Generate final research report combining all research components.
        
        Args:
            query: Original research query
            search_results: Results from paper search
            analysis: Analysis results from analysis agent
            report_type: Type of report (executive, comprehensive, technical, brief)
            
        Returns:
            Complete formatted research report
        """
        if report_type == "final":
            return summary_agent.format_final_report(query, search_results, analysis)
        else:
            combined_data = f"Query: {query}\n\nSearch Results:\n{search_results}\n\nAnalysis:\n{analysis}"
            return summary_agent.generate_research_report(combined_data, report_type)
    
    return final_report_tool


def create_literature_review_tool():
    """Create literature review tool."""
    
    summary_agent = SummaryAgent()
    
    @tool("create_literature_review")
    def literature_review_tool(papers_data: str, focus_area: str = "") -> str:
        """
        Create a comprehensive literature review from research papers.
        
        Args:
            papers_data: Data about research papers
            focus_area: Specific area to focus the review on
            
        Returns:
            Formatted literature review
        """
        return summary_agent.create_literature_review(papers_data, focus_area)
    
    return literature_review_tool