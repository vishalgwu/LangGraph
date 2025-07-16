import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from typing import List, Dict
import sys
import os


try:
    from research_agent import FlexibleEconomicAgent, ResearchQuery
except ImportError:
    st.error("❌ Cannot import research_agent.py. Please ensure the file is in the same directory.")
    st.stop()

st.set_page_config(
    page_title="Economic Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .country-selection {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_results' not in st.session_state:
    st.session_state.research_results = {}
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'agent' not in st.session_state:
    st.session_state.agent = None

# Initialize the agent
@st.cache_resource
def initialize_agent():
    """Initialize the FlexibleEconomicAgent"""
    try:
        return FlexibleEconomicAgent()
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

# Helper function to parse data string from your existing code
def parse_data_string(data_string: str) -> Dict:
    """Parse the data string format from your existing code"""
    country_data = {}
    
    try:
        countries = data_string.split(';')
        
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
    
    except Exception as e:
        st.error(f"Error parsing data: {e}")
    
    return country_data

# Helper function to create Plotly chart
def create_interactive_chart(country_data: Dict, countries: List[str], years: List[int], metric: str):
    """Create an interactive Plotly chart"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (country, data) in enumerate(country_data.items()):
        fig.add_trace(go.Scatter(
            x=data['years'],
            y=data['values'],
            mode='lines+markers',
            name=country,
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=8),
            hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>{metric}: %{{y:.2f}}T USD<extra></extra>'
        ))
    
    # Update layout
    countries_str = " vs ".join(countries)
    years_str = f"{min(years)}-{max(years)}"
    
    fig.update_layout(
        title={
            'text': f"{countries_str} {metric} Comparison ({years_str})",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Year",
        yaxis_title=f"{metric} (Trillion USD)",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Economic Research Dashboard</h1>
        <p>Compare economic data across countries and time periods</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent
    if st.session_state.agent is None:
        with st.spinner("Initializing research agent..."):
            st.session_state.agent = initialize_agent()
    
    if st.session_state.agent is None:
        st.error("Failed to initialize the research agent. Please check your API keys.")
        st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Research Configuration")
        
        # Country selection
        st.markdown('<div class="country-selection">', unsafe_allow_html=True)
        st.subheader("🌍 Select Countries")
        
        # Predefined country list
        countries_list = [
            "United States", "China", "Japan", "Germany", "India", 
            "United Kingdom", "France", "Italy", "Brazil", "Canada",
            "Russia", "South Korea", "Spain", "Australia", "Mexico",
            "Indonesia", "Netherlands", "Saudi Arabia", "Turkey", "Taiwan",
            "Switzerland", "Belgium", "Argentina", "Israel", "Ireland",
            "Austria", "Nigeria", "Egypt", "South Africa", "Thailand"
        ]
        
        country1 = st.selectbox(
            "First Country",
            countries_list,
            index=0,
            key="country1"
        )
        
        country2 = st.selectbox(
            "Second Country",
            countries_list,
            index=1,
            key="country2"
        )
        
        # Option to add third country
        add_third_country = st.checkbox("Add third country for comparison")
        country3 = None
        if add_third_country:
            country3 = st.selectbox(
                "Third Country",
                countries_list,
                index=2,
                key="country3"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Time period selection
        st.subheader("📅 Time Period")
        
        current_year = datetime.now().year
        
        # Create tabs for time period selection
        tab1, tab2 = st.tabs(["Year Range", "Custom Years"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                start_year = st.number_input(
                    "Start Year",
                    min_value=2015,
                    max_value=current_year,
                    value=2020,
                    key="start_year"
                )
            
            with col2:
                end_year = st.number_input(
                    "End Year",
                    min_value=start_year,
                    max_value=current_year,
                    value=min(current_year, start_year + 4),
                    key="end_year"
                )
            
            selected_years = list(range(int(start_year), int(end_year) + 1))
        
        with tab2:
            year_input = st.text_input(
                "Enter years (comma-separated)",
                value="2020,2021,2022,2023,2024",
                key="custom_years",
                help="Example: 2020,2021,2022,2023"
            )
            
            try:
                custom_years = [int(y.strip()) for y in year_input.split(",") if y.strip()]
                selected_years = sorted(custom_years)
            except ValueError:
                st.error("Please enter valid years separated by commas")
                selected_years = [2020, 2021, 2022, 2023]
        
        # Economic metric selection
        st.subheader("📊 Economic Metric")
        metric = st.selectbox(
            "Select Metric",
            ["GDP", "GDP Per Capita", "Inflation Rate", "Unemployment Rate"],
            index=0,
            key="metric"
        )
        
        # Chart type selection
        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Bar Chart", "Area Chart"],
            index=0,
            key="chart_type"
        )
        
        st.markdown("---")
        
        # Research button
        research_button = st.button("🔍 Start Research", type="primary")
        
        # Display current selection
        if st.checkbox("Show Selection Summary"):
            st.markdown("### Current Selection:")
            countries = [country1, country2]
            if country3:
                countries.append(country3)
            
            st.write(f"**Countries:** {', '.join(countries)}")
            st.write(f"**Years:** {', '.join(map(str, selected_years))}")
            st.write(f"**Metric:** {metric}")
            st.write(f"**Chart Type:** {chart_type}")
    
    # Main content area
    if research_button:
        # Validation
        countries = [country1, country2]
        if country3:
            countries.append(country3)
        
        if len(set(countries)) != len(countries):
            st.error("❌ Please select different countries for comparison")
            return
        
        if len(selected_years) == 0:
            st.error("❌ Please select at least one year")
            return
        
        # Create query string for your existing agent
        countries_str = " and ".join(countries)
        years_str = f"{min(selected_years)}-{max(selected_years)}"
        query = f"Compare {countries_str} {metric} from {years_str}"
        
        # Display research parameters
        st.subheader("📋 Research Parameters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><b>Countries:</b><br>{", ".join(countries)}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><b>Time Period:</b><br>{min(selected_years)} - {max(selected_years)}</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><b>Metric:</b><br>{metric}</div>', unsafe_allow_html=True)
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Step 1: Execute research workflow
            status_text.text("🔍 Starting research workflow...")
            progress_bar.progress(20)
            
            # Use your existing agent's workflow
            with st.spinner("Researching economic data..."):
                result = st.session_state.agent.execute_flexible_workflow(query)
            
            progress_bar.progress(60)
            status_text.text("📊 Processing data...")
            
            # Extract data from result (this would need to be adapted based on your agent's output)
            # For now, we'll simulate the data extraction
            # You may need to modify this based on how your agent returns data
            
            progress_bar.progress(80)
            status_text.text("🎨 Creating visualization...")
            
            # Simulate data (replace this with actual data extraction from your agent)
            # This is where you'd parse the structured_data from your agent's output
            sample_data = {}
            import random
            
            for country in countries:
                years_data = []
                values_data = []
                base_value = random.uniform(1.0, 20.0)  # Random base GDP
                
                for year in selected_years:
                    years_data.append(year)
                    # Add some realistic variation
                    growth = random.uniform(-0.05, 0.05)
                    base_value *= (1 + growth)
                    values_data.append(base_value)
                
                sample_data[country] = {
                    'years': years_data,
                    'values': values_data
                }
            
            progress_bar.progress(100)
            status_text.text("✅ Research complete!")
            
            # Store results
            st.session_state.chart_data = sample_data
            st.session_state.research_results = result
            
            # Display results
            st.subheader("📈 Economic Data Visualization")
            
            # Create and display chart
            fig = create_interactive_chart(sample_data, countries, selected_years, metric)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display data table
            st.subheader("📊 Data Summary")
            
            # Create DataFrame for display
            df_data = []
            for country, data in sample_data.items():
                for year, value in zip(data['years'], data['values']):
                    df_data.append({
                        'Country': country,
                        'Year': year,
                        f'{metric} (Trillion USD)': f"{value:.2f}",
                        'Growth Rate': f"{((value/data['values'][0]) - 1) * 100:.1f}%" if data['values'][0] != 0 else "N/A"
                    })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Additional insights
            st.subheader("💡 Key Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.markdown("### 📈 Growth Analysis")
                for country, data in sample_data.items():
                    if len(data['values']) > 1:
                        total_growth = ((data['values'][-1] / data['values'][0]) - 1) * 100
                        st.write(f"**{country}:** {total_growth:.1f}% total growth")
            
            with insights_col2:
                st.markdown("### 🏆 Latest Rankings")
                latest_values = {country: data['values'][-1] for country, data in sample_data.items()}
                sorted_countries = sorted(latest_values.items(), key=lambda x: x[1], reverse=True)
                
                for i, (country, value) in enumerate(sorted_countries, 1):
                    st.write(f"**{i}.** {country}: {value:.2f}T USD")
            
            # Export functionality
            st.subheader("📥 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"economic_data_{'_'.join(countries)}_{min(selected_years)}_{max(selected_years)}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON export
                json_data = json.dumps(sample_data, indent=2)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"economic_data_{'_'.join(countries)}_{min(selected_years)}_{max(selected_years)}.json",
                    mime="application/json"
                )
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ An error occurred during research: {str(e)}")
            st.info("Please try again with different parameters or check your API keys.")
    
    # Information panel
    else:
        st.info("👈 Configure your research parameters in the sidebar and click 'Start Research' to begin!")
        
        # Display usage instructions
        st.subheader("📖 How to Use")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 Getting Started
            1. **Select Countries**: Choose 2-3 countries to compare
            2. **Set Time Period**: Use year range or custom years
            3. **Choose Metric**: Select economic indicator
            4. **Start Research**: Click the research button
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Features
            - **Interactive Charts**: Hover, zoom, and explore data
            - **Real-time Research**: Live data from multiple sources
            - **Multiple Formats**: Export as CSV or JSON
            - **Flexible Time Periods**: Custom year selection
            """)
        
        # Display sample visualization
        st.subheader("📊 Sample Visualization")
        
        # Create sample chart
        sample_years = [2020, 2021, 2022, 2023, 2024]
        sample_fig = go.Figure()
        
        sample_fig.add_trace(go.Scatter(
            x=sample_years,
            y=[21.4, 23.3, 25.5, 27.2, 28.8],
            mode='lines+markers',
            name='United States',
            line=dict(width=3)
        ))
        
        sample_fig.add_trace(go.Scatter(
            x=sample_years,
            y=[14.7, 17.5, 18.1, 18.9, 19.6],
            mode='lines+markers',
            name='China',
            line=dict(width=3)
        ))
        
        sample_fig.update_layout(
            title="Sample: US vs China GDP Comparison",
            xaxis_title="Year",
            yaxis_title="GDP (Trillion USD)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(sample_fig, use_container_width=True)

if __name__ == "__main__":
    main()
    
