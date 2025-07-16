import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Dict, List, Any, Optional
import json
import matplotlib.pyplot as plt
import pandas as pd
from serpapi import GoogleSearch
import re
from pydantic import BaseModel

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo")



class CountryEconomicData(BaseModel):
    country: str
    year: int
    gdp_value: float
    gdp_unit: str
    currency: str = "USD"
    source: str = ""

class ResearchQuery(BaseModel):
    countries: List[str]
    years: List[int]
    metric: str = "GDP"
    chart_type: str = "line"

@tool
def flexible_economic_research(query: str) -> str:
    """Research economic data for any country and time period"""
    try:
        
        search_queries = [
            f"{query} GDP statistics",
            f"{query} economic data",
            f"{query} gross domestic product"
        ]
        
        all_results = []
        
        for search_query in search_queries[:2]:  
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": SERPAPI_API_KEY,
                "num": 3
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            for result in results.get("organic_results", []):
                all_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", ""),
                    "search_query": search_query
                })
        
        return json.dumps(all_results, indent=2)
    
    except Exception as e:
        return f"Research failed: {str(e)}"

@tool
def create_flexible_chart(chart_data: str, chart_config: str) -> str:
    """Create various types of charts from economic data
    
    chart_data format: country1|year1:value1,year2:value2;country2|year1:value1,year2:value2
    chart_config format: {"type": "line/bar/comparison", "title": "Custom Title", "ylabel": "GDP (Trillion USD)"}
    """
    try:
        
        config = json.loads(chart_config)
        chart_type = config.get("type", "line")
        title = config.get("title", "Economic Data Chart")
        ylabel = config.get("ylabel", "GDP (Trillion USD)")
        
        
        country_data = {}
        countries = chart_data.split(';')
        
        for country_entry in countries:
            if '|' not in country_entry:
                continue
                
            country, data_points = country_entry.split('|')
            country = country.strip()
            
            years = []
            values = []
            
            for point in data_points.split(','):
                if ':' in point:
                    year, value = point.split(':')
                    years.append(int(year.strip()))
                    values.append(float(value.strip()))
            
            country_data[country] = {'years': years, 'values': values}
        
        
        plt.figure(figsize=(12, 8))
        
        if chart_type == "line":
            for country, data in country_data.items():
                plt.plot(data['years'], data['values'], marker='o', linewidth=2, 
                        markersize=8, label=country)
        
        elif chart_type == "bar":
            
            countries_list = list(country_data.keys())
            latest_values = [country_data[country]['values'][-1] for country in countries_list]
            plt.bar(countries_list, latest_values, alpha=0.7)
        
        elif chart_type == "comparison":
            
            years_set = set()
            for data in country_data.values():
                years_set.update(data['years'])
            years_list = sorted(years_set)
            
            x = range(len(years_list))
            width = 0.35
            
            for i, (country, data) in enumerate(country_data.items()):
                values_aligned = []
                for year in years_list:
                    if year in data['years']:
                        idx = data['years'].index(year)
                        values_aligned.append(data['values'][idx])
                    else:
                        values_aligned.append(0)
                
                plt.bar([xi + i * width for xi in x], values_aligned, 
                       width, label=country, alpha=0.7)
            
            plt.xticks([xi + width/2 for xi in x], years_list)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        
        filename = f"economic_chart_{chart_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return f"Chart created successfully: {filename}. Countries: {list(country_data.keys())}, Type: {chart_type}"
    
    except Exception as e:
        return f"Chart creation failed: {str(e)}"

class FlexibleEconomicAgent:
    def __init__(self):
        self.llm = llm
        self.research_tool = flexible_economic_research
        self.chart_tool = create_flexible_chart
    
    def parse_user_query(self, query: str) -> ResearchQuery:
        """Parse user query to extract countries, years, and requirements"""
        
        parsing_prompt = f"""
        Parse this economic data request and extract:
        1. Countries mentioned (convert to standard names)
        2. Years or time periods
        3. Type of economic data (GDP, inflation, etc.)
        4. Chart type if specified
        
        Query: "{query}"
        
        Return as JSON:
        {{
            "countries": ["Country1", "Country2"],
            "years": [2021, 2022, 2023],
            "metric": "GDP",
            "chart_type": "line"
        }}
        
        Common country mappings:
        - UK/Britain -> United Kingdom
        - US/USA/America -> United States
        - China/PRC -> China
        - India -> India
        """
        
        response = self.llm.invoke([HumanMessage(content=parsing_prompt)])
        
        try:
            parsed = json.loads(response.content)
            return ResearchQuery(**parsed)
        except:
            
            countries = []
            years = []
            
            
            country_patterns = {
                r'\buk\b|\bbritain\b|\bunited kingdom\b': 'United Kingdom',
                r'\busa\b|\bus\b|\bamerica\b|\bunited states\b': 'United States',
                r'\bchina\b': 'China',
                r'\bindia\b': 'India',
                r'\bjapan\b': 'Japan',
                r'\bgermany\b': 'Germany',
                r'\bfrance\b': 'France'
            }
            
            query_lower = query.lower()
            for pattern, country in country_patterns.items():
                if re.search(pattern, query_lower):
                    countries.append(country)
            
            # Extract years
            year_matches = re.findall(r'\b(20\d{2})\b', query)
            years = [int(year) for year in year_matches]
            
            if not years:
                years = [2022, 2023, 2024]  # Default recent years
            
            return ResearchQuery(
                countries=countries or ['United Kingdom'],
                years=years,
                metric="GDP",
                chart_type="line"
            )
    
    def research_economic_data(self, research_query: ResearchQuery) -> str:
        """Research economic data for specified countries and years"""
        
        research_results = {}
        
        for country in research_query.countries:
            print(f"🔍 Researching {country} {research_query.metric} data...")
            
            # Create targeted search query
            years_str = " ".join(map(str, research_query.years))
            search_query = f"{country} {research_query.metric} {years_str}"
            
            result = self.research_tool.invoke({"query": search_query})
            research_results[country] = result
        
        return json.dumps(research_results, indent=2)
    
    def extract_structured_data(self, research_results: str, research_query: ResearchQuery) -> str:
        """Extract structured data from research results"""
        
        extraction_prompt = f"""
        Extract economic data from these research results and format for chart creation.
        
        Target format: country1|year1:value1,year2:value2;country2|year1:value1,year2:value2
        
        Countries: {research_query.countries}
        Years: {research_query.years}
        Metric: {research_query.metric}
        
        Research Results:
        {research_results}
        
        Instructions:
        1. Extract GDP values in trillion USD (convert if needed)
        2. If exact values not found, use reasonable estimates
        3. Ensure all countries have data for requested years
        4. Use format: UK|2022:2.3,2023:2.4,2024:2.5;China|2022:17.8,2023:18.1,2024:18.5
        
        Return ONLY the formatted data string.
        """
        
        response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
        return response.content.strip()
    
    def create_visualization(self, data_string: str, research_query: ResearchQuery) -> str:
        """Create appropriate visualization"""
        
        # Generate chart title
        countries_str = " vs ".join(research_query.countries)
        years_str = f"{min(research_query.years)}-{max(research_query.years)}"
        
        chart_config = {
            "type": research_query.chart_type,
            "title": f"{countries_str} {research_query.metric} ({years_str})",
            "ylabel": f"{research_query.metric} (Trillion USD)"
        }
        
        result = self.chart_tool.invoke({
            "chart_data": data_string,
            "chart_config": json.dumps(chart_config)
        })
        
        return result
    
    def execute_flexible_workflow(self, user_query: str) -> str:
        """Main workflow that handles any economic research query"""
        
        print("🎯 Starting flexible economic research workflow...")
        print(f"📝 User Query: {user_query}")
        
       
        research_query = self.parse_user_query(user_query)
        print(f"📊 Parsed Request: {research_query.countries} | {research_query.years} | {research_query.metric}")
        
        
        research_results = self.research_economic_data(research_query)
        
        # Step 3: Extract structured data
        print("🔄 Extracting structured data...")
        structured_data = self.extract_structured_data(research_results, research_query)
        print(f"📈 Structured Data: {structured_data}")
        
        # Step 4: Create visualization
        print("🎨 Creating visualization...")
        chart_result = self.create_visualization(structured_data, research_query)
        
        return f"""
        ✅ Workflow Complete!
        
        📊 Research Query: {research_query.countries} {research_query.metric} for {research_query.years}
        📈 Data Extracted: {structured_data}
        🎨 Chart Result: {chart_result}
        """



def run_examples():
    agent = FlexibleEconomicAgent()
    
    test_queries = [
        "Get UK GDP for the last 3 years and make a line chart",
        "Compare China and India GDP from 2020 to 2024",
        "Show me Ukraine GDP data for 2022 and 2023 with a bar chart",
        "Compare US, China, and India economic growth 2021-2024",
        "Get Japan GDP data and create a visualization"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}: {query}")
        print('='*60)
        
        try:
            result = agent.execute_flexible_workflow(query)
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Interactive mode
    agent = FlexibleEconomicAgent()
    
    print("🌍 Flexible Economic Research Agent")
    print("Ask me about any country's economic data!")
    print("Examples:")
    print("- 'Compare UK and France GDP 2020-2024'")
    print("- 'Show China economic growth last 5 years'")
    print("- 'Ukraine GDP 2022 vs 2023 bar chart'")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n💬 Your query (or 'quit' to exit): ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'examples':
                run_examples()
                continue
            
            result = agent.execute_flexible_workflow(user_input)
            print(result)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again with a different query.")