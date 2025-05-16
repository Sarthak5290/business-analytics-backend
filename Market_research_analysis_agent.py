# !pip install -qU langchain_groq langchain_google_genai langchain_community langgraph reportlab markdown
## Install these packages before running

import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from flask import Flask, request, send_file, make_response, jsonify
import os
import re
from typing import TypedDict, Annotated, Sequence
import operator
import csv
import logging
import json

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import chain
# from langchain_google_genai import ChatGoogleGenerativeAI  # No longer used
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END

# PDF Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import markdown

app = Flask(__name__)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM Setup ---
GROQ_API_KEY = ""
if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY not found in environment variables.")
    raise ValueError("GROQ_API_KEY must be set in environment variables.")

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.0,
    max_retries=2,
    groq_api_key=GROQ_API_KEY,
)

# --- Search Tool Setup (Google CSE) ---
GOOGLE_API_KEY = ""
GOOGLE_CSE_ID = ""

if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    logging.error("GOOGLE_API_KEY and GOOGLE_CSE_ID not found in environment variables.")
    raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in environment.")

search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

# --- State Definition ---
class AgentState(TypedDict):
    input: str
    user_idea: str
    detected_industry: str | None
    intermediate_steps: Annotated[list[tuple[AIMessage, str]], operator.add]
    agent_outcome: BaseMessage | None
    competitor_analysis: str | None
    market_trends: str | None
    swot_analysis: str | None
    comparison: str | None
    numerical_data: str | None

# --- Node Functions ---
def detect_industry(state):
    user_idea = state["user_idea"]
    logging.info(f"Detecting industry for: {user_idea}")
    query = f"What industry is {user_idea} in?"
    search_result = search.run(query)
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Extract the primary industry from the search results. Output *only* the industry name."),
      ("user", "Search Results:\n{search_results}\n\nIndustry:")
    ])
    extraction_chain = prompt | llm | StrOutputParser()
    industry = extraction_chain.invoke({"search_results": search_result})
    logging.info(f"Detected industry: {industry}")
    return {"detected_industry": industry}

def analyze_competitors(state):
    industry = state["detected_industry"]
    logging.info(f"Analyzing competitors for industry: {industry}")
    query = f"top competitors in the {industry} industry"
    search_results = search.run(query)
    prompt = ChatPromptTemplate.from_messages([
      ("system", "List top competitors in {industry}. Summarize strengths/weaknesses. Numbered list. Return in Python list format."),
      ("user", "Search Results:\n{search_results}")
    ])
    analysis_chain = prompt | llm | StrOutputParser()
    competitor_analysis = analysis_chain.invoke({"industry": industry, "search_results": search_results})
    logging.info("Competitor analysis complete.")
    return {"competitor_analysis": competitor_analysis}

def get_market_trends(state):
    industry = state["detected_industry"]
    logging.info(f"Getting market trends for industry: {industry}")
    query = f"latest market trends in the {industry} industry"
    search_results = search.run(query)
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Summarize latest market trends in {industry} from search results."),
      ("user", "Search Results:\n{search_results}")
    ])
    trends_chain = prompt | llm | StrOutputParser()
    market_trends = trends_chain.invoke({"industry": industry, "search_results": search_results})
    logging.info("Market trends retrieved.")
    return {"market_trends": market_trends}

def generate_swot(state):
    industry = state["detected_industry"]
    logging.info(f"Generating SWOT analysis for industry: {industry}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a concise SWOT analysis for a startup in {industry}. List strengths, weaknesses, opportunities, and threats with bullet points."),
        ("user", "Industry: {industry}")
    ])
    swot_chain = prompt | llm | StrOutputParser()
    swot_analysis = swot_chain.invoke({"industry": industry})
    logging.info("SWOT analysis generated.")
    return {"swot_analysis": swot_analysis}

def get_numerical_market_data(state):
    industry = state["detected_industry"]
    logging.info(f"Getting numerical market data for industry: {industry}")
    query = f"market size, growth rate, and key statistics for {industry} industry"
    search_results = search.run(query)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract numerical data (market size, growth rate, etc.) for {industry} from the search results. Present the data in a clear, concise, and organized format, using bullet points or a table. Cite the source snippet briefly. CSV format."),
        ("user", "Search Results:\n{search_results}")
    ])
    data_chain = prompt | llm | StrOutputParser()
    numerical_data = data_chain.invoke({"industry": industry, "search_results": search_results})
    logging.info("Numerical market data retrieved.")
    return {"numerical_data": numerical_data}

def compare_idea(state):
    user_idea = state["user_idea"]
    competitor_analysis = state["competitor_analysis"]
    logging.info(f"Comparing idea '{user_idea}' with competitors.")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Compare the business idea to competitors, highlighting advantages. Be specific."),
        ("user", "Business Idea: {user_idea}\n\nCompetitor Analysis:\n{competitor_analysis}")
    ])
    comparison_chain = prompt | llm | StrOutputParser()
    comparison = comparison_chain.invoke({"user_idea": user_idea, "competitor_analysis": competitor_analysis})
    logging.info("Idea comparison complete.")
    return {"comparison": comparison, "agent_outcome": AIMessage(content="Analysis complete.")}

def should_continue(state):
    return "end" if state["agent_outcome"] is not None else "continue"

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("detect_industry", detect_industry)
workflow.add_node("analyze_competitors", analyze_competitors)
workflow.add_node("get_market_trends", get_market_trends)
workflow.add_node("generate_swot", generate_swot)
workflow.add_node("get_numerical_market_data", get_numerical_market_data)
workflow.add_node("compare_idea", compare_idea)
workflow.set_entry_point("detect_industry")
workflow.add_edge("detect_industry", "analyze_competitors")
workflow.add_edge("analyze_competitors", "get_market_trends")
workflow.add_edge("get_market_trends", "generate_swot")
workflow.add_edge("generate_swot", "get_numerical_market_data")
workflow.add_edge("get_numerical_market_data", "compare_idea")
workflow.add_conditional_edges("compare_idea", should_continue, {"continue": END, "end": END})
graph = workflow.compile()

# --- Enhanced PDF Generation ---
def generate_pdf_report(result: dict, output_pdf_path: str):
    """Generates a structured PDF report from the analysis results."""
    logging.info(f"Generating PDF report at: {output_pdf_path}")
    
    doc = SimpleDocTemplate(
        output_pdf_path, 
        pagesize=letter,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=18
    )
    styles = getSampleStyleSheet()
    # Create a custom style for bullet items with extra space.
    bullet_style = ParagraphStyle('BulletStyle', parent=styles['Normal'], spaceAfter=6)
    
    story = []
    
    # --- Title Page ---
    title = Paragraph("Business Idea Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.5*inch))
    
    subtitle = Paragraph("Detailed Analysis Report", styles['Heading2'])
    story.append(subtitle)
    story.append(Spacer(1, 0.5*inch))
    
    # --- Helper function to add sections ---
    def add_section(section_title, content):
        story.append(Paragraph(section_title, styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # If this is the SWOT Analysis section, render it as a graphic table.
        if section_title.lower() == "swot analysis":
            # Attempt to parse the SWOT content using markers.
            strengths = re.findall(r"Strengths:\s*(.*?)\s*(Weaknesses:|Opportunities:|Threats:|$)", content, re.DOTALL)
            weaknesses = re.findall(r"Weaknesses:\s*(.*?)\s*(Opportunities:|Threats:|$)", content, re.DOTALL)
            opportunities = re.findall(r"Opportunities:\s*(.*?)\s*(Threats:|$)", content, re.DOTALL)
            threats = re.findall(r"Threats:\s*(.*)", content, re.DOTALL)
            
            # Use the first match from each if available.
            strengths_text = strengths[0][0].strip() if strengths else "N/A"
            weaknesses_text = weaknesses[0][0].strip() if weaknesses else "N/A"
            opportunities_text = opportunities[0][0].strip() if opportunities else "N/A"
            threats_text = threats[0].strip() if threats else "N/A"
            
            # Create a table with 2 columns for strengths/weaknesses and opportunities/threats.
            swot_data = [
                [Paragraph("<b>Strengths</b>", styles['Heading3']), Paragraph("<b>Weaknesses</b>", styles['Heading3'])],
                [Paragraph(strengths_text, styles['Normal']), Paragraph(weaknesses_text, styles['Normal'])],
                [Paragraph("<b>Opportunities</b>", styles['Heading3']), Paragraph("<b>Threats</b>", styles['Heading3'])],
                [Paragraph(opportunities_text, styles['Normal']), Paragraph(threats_text, styles['Normal'])]
            ]
            t = Table(swot_data, colWidths=[250, 250])
            t.setStyle(TableStyle([
                ('BOX', (0,0), (-1,-1), 1, 'black'),
                ('INNERGRID', (0,0), (-1,-1), 0.5, 'grey'),
                ('BACKGROUND', (0,0), (-1,0), '#d3d3d3'),
                ('BACKGROUND', (0,2), (-1,2), '#d3d3d3'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.4*inch))
            return
        
        # If content is valid JSON (for competitor analysis), format as table.
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                if section_title.lower() == "competitor analysis":
                    table_data = [["Name", "Strengths", "Weaknesses"]]
                    for comp in parsed:
                        row = [comp.get("name", ""), comp.get("strengths", ""), comp.get("weaknesses", "")]
                        table_data.append(row)
                    t = Table(table_data, colWidths=[150, 200, 150])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), '#d3d3d3'),
                        ('TEXTCOLOR', (0,0), (-1,0), 'black'),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 10),
                        ('BOTTOMPADDING', (0,0), (-1,0), 6),
                        ('GRID', (0,0), (-1,-1), 0.5, 'black'),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 0.4*inch))
                    return
        except Exception as e:
            pass
        
        # For Numerical Market Data, add a colored box/table for emphasis.
        if section_title.lower() == "numerical market data":
            t = Table([[Paragraph(content, styles['Normal'])]], colWidths=[500])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), '#f0f8ff'),
                ('BOX', (0,0), (-1,-1), 1, 'black'),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.4*inch))
            return
        
        # If content contains bullet markers (- or *), split and add with extra spacing.
        if re.search(r"^[-*]\s+", content, re.MULTILINE):
            bullet_lines = content.splitlines()
            bullet_items = []
            for line in bullet_lines:
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    bullet_text = line[2:].strip()
                    bullet_items.append(ListItem(Paragraph(bullet_text, bullet_style), bulletColor='black'))
                elif line:
                    bullet_items.append(ListItem(Paragraph(line, bullet_style), bulletColor='black'))
            if bullet_items:
                bullet_list = ListFlowable(
                    bullet_items, 
                    bulletType='bullet', 
                    start='circle', 
                    bulletFontName='Helvetica'
                )
                story.append(bullet_list)
        else:
            html_content = markdown.markdown(content)
            story.append(Paragraph(html_content, styles['Normal']))
        story.append(Spacer(1, 0.4*inch))
    
    # --- Add sections from the analysis results ---
    add_section("Competitor Analysis", result.get('competitor_analysis', 'No data available.'))
    add_section("Market Trends", result.get('market_trends', 'No data available.'))
    add_section("SWOT Analysis", result.get('swot_analysis', 'No data available.'))
    add_section("Comparison with Competitors", result.get('comparison', 'No data available.'))
    add_section("Numerical Market Data", result.get('numerical_data', 'No data available.'))
    
    # --- Extract and add Sources ---
    sources_set = set()
    for key in ['competitor_analysis', 'market_trends', 'swot_analysis', 'comparison', 'numerical_data']:
        text = result.get(key, "")
        # Extract URLs using regex.
        urls = re.findall(r'(https?://[^\s\]]+)', text)
        for url in urls:
            sources_set.add(url)
    if sources_set:
        story.append(Paragraph("Sources", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        for url in sorted(sources_set):
            # Create clickable link using HTML anchor tags.
            link = f'<a href="{url}" color="blue">{url}</a>'
            story.append(Paragraph(link, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    
    # Build and save the PDF document.
    doc.build(story)
    logging.info("PDF report generated successfully.")

def run_analysis(user_idea: str, llm):
    """Runs the full analysis and returns output file paths."""
    logging.info(f"Starting analysis for user idea: {user_idea}")

    inputs = {"input": user_idea, "user_idea": user_idea}
    result = graph.invoke(inputs)

    # --- Company extraction logic ---
    compnies_names_output = llm.invoke(result['competitor_analysis'] + " Give me python list of companies mentioned")
    pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
    extracted_companies_robust = re.findall(pattern, compnies_names_output.content)
    logging.info(f"Extracted companies: {extracted_companies_robust}")

    # --- Reinitialize LLM and Search Tool if needed ---
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.0,
        max_retries=2,
        groq_api_key=GROQ_API_KEY
    )
    search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]

    # --- State Definition for Company Analysis ---
    class CompanyAnalysisState(TypedDict):
        company_name: str
        market_analysis: str | None

    def analyze_company_market(state: CompanyAnalysisState) -> CompanyAnalysisState:
        company_name = state["company_name"]
        logging.info(f"Analyzing market for: {company_name}")
        query = f"{company_name} market analysis"
        try:
            search_results = search.run(query)
            prompt = ChatPromptTemplate.from_messages([
              ("system", "Provide a concise market analysis for {company_name}, including key trends, competitors, and market size/growth if available. Format as markdown with bullet points."),
              ("user", "Search Results:\n{search_results}")
            ])
            analysis_chain = prompt | llm | StrOutputParser()
            market_analysis = analysis_chain.invoke({"company_name": company_name, "search_results": search_results})
            logging.info(f"Market analysis generated for {company_name}")
            return {"market_analysis": market_analysis}
        except Exception as e:
            logging.error(f"Error during market analysis for {company_name}: {e}")
            return {"market_analysis": f"Error: Could not retrieve market analysis. Details: {e}"}

    def should_continue(state):
        return "end"

    def create_company_graph():
        workflow = StateGraph(CompanyAnalysisState)
        workflow.add_node("analyze_market", analyze_company_market)
        workflow.set_entry_point("analyze_market")
        workflow.add_conditional_edges("analyze_market", should_continue, {"continue": END, "end": END})
        return workflow.compile()

    def analyze_companies(companies: list, output_csv_path: str):
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Company Name', 'Market Analysis']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for company_name in companies:
                company_graph = create_company_graph()
                inputs = {"company_name": company_name}
                try:
                    result = company_graph.invoke(inputs)
                    market_analysis = result.get("market_analysis", "No analysis available.")
                    writer.writerow({
                        'Company Name': company_name,
                        'Market Analysis': market_analysis
                    })
                    logging.info(f"Processed: {company_name}")
                except Exception as e:
                    logging.error(f"Error processing {company_name}: {e}")
                    writer.writerow({
                        'Company Name': company_name,
                        'Market Analysis': f"Error: {e}"
                    })

    output_csv_path = "company_market_analysis.csv"
    analyze_companies(extracted_companies_robust, output_csv_path)
    logging.info(f"CSV report saved to: {output_csv_path}")

    output_pdf_path = "business_idea_analysis.pdf"
    generate_pdf_report(result, output_pdf_path)

    return output_csv_path, output_pdf_path

# --- Flask Route ---
@app.route('/', methods=['POST'])
def analyze():
    logging.info("Received request with JSON input.")
    try:
        data = request.get_json()
        if not data or "user_idea" not in data:
            logging.error("Missing 'user_idea' in JSON input.")
            return jsonify({"error": "Missing 'user_idea' in JSON input."}), 400

        user_idea = data["user_idea"]
        logging.info("User idea extracted from JSON input.")

        csv_path, pdf_path = run_analysis(user_idea, llm)

        # Create a response that sends both CSV and PDF as attachments.
        response = make_response()
        response.headers['Content-Disposition'] = 'attachment; filename=results.zip'
        response.headers['Content-Type'] = 'application/zip'

        csv_response = send_file(csv_path, as_attachment=True, download_name="company_analysis.csv")
        pdf_response = send_file(pdf_path, as_attachment=True, download_name="business_analysis.pdf")

        return [csv_response, pdf_response]

    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
