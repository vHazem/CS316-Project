# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Advanced libraries - comment out if causing issues
# import shap
# import joblib

# Page configuration
st.set_page_config(
    page_title="Food Waste Analytics Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark mode friendly colors
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2196F3;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: rgba(45, 45, 45, 0.7);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2196F3;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        text-align: center;
        color: #b0bec5;
    }
    .insight-card {
        background-color: rgba(46, 125, 50, 0.2);
        border-left: 8px solid #4CAF50;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .insight-title {
        font-weight: 600;
        color: #4CAF50;
        font-size: 1.2rem;
        margin-bottom: 8px;
    }
    .insight-value {
        font-weight: 700;
        color: #69F0AE;
        font-size: 1.4rem;
    }
    .warning-card {
        background-color: rgba(245, 124, 0, 0.2);
        border-left: 8px solid #F57C00;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .feature-importance {
        background-color: rgba(66, 66, 66, 0.7);
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #b0bec5;
        border-top: 1px solid #424242;
        padding-top: 20px;
    }
    .explanation {
        background-color: rgba(25, 118, 210, 0.2);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 8px solid #2196F3;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .explanation h3 {
        color: #42A5F5;
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 1.4rem;
        border-bottom: 2px solid rgba(33, 150, 243, 0.3);
        padding-bottom: 8px;
    }
    .explanation p, .explanation ul, .explanation li {
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .explanation strong {
        color: #64B5F6;
    }
    .recommendation-card {
        background-color: rgba(255, 152, 0, 0.2);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 8px solid #FF9800;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .recommendation-card h3 {
        color: #FFB74D;
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 1.4rem;
        border-bottom: 2px solid rgba(255, 152, 0, 0.3);
        padding-bottom: 8px;
    }
    .recommendation-card li {
        margin-bottom: 10px;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .recommendation-card strong {
        color: #FFA726;
    }
    .feature-importance-card {
        background-color: rgba(156, 39, 176, 0.2);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 8px solid #9C27B0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .feature-importance-card h3 {
        color: #BA68C8;
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 1.4rem;
        border-bottom: 2px solid rgba(156, 39, 176, 0.3);
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("merged_data.csv")
    return data



# Function to train multiple models
@st.cache_resource
def train_models(X, y):
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # For univariate models, select only first feature
    X_train_uni = X_train.iloc[:, 0].values.reshape(-1, 1)
    X_test_uni = X_test.iloc[:, 0].values.reshape(-1, 1)
    
    # 1. Linear model (univariate)
    linear_model_uni = LinearRegression()
    linear_model_uni.fit(X_train_uni, y_train)
    
    # 2. Non-linear model - SVR (univariate)
    svr_model_uni = SVR(kernel='rbf')
    # Scale data for SVR
    scaler_uni = StandardScaler()
    X_train_uni_scaled = scaler_uni.fit_transform(X_train_uni)
    X_test_uni_scaled = scaler_uni.transform(X_test_uni)
    svr_model_uni.fit(X_train_uni_scaled, y_train)
    
    # 3. Multivariate model - Random Forest
    rf_model_multi = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_multi.fit(X_train, y_train)
    
    # Evaluate models
    results = {}
    
    # Evaluate linear univariate
    y_pred_linear_uni = linear_model_uni.predict(X_test_uni)
    results["linear_uni"] = {
        "r2": r2_score(y_test, y_pred_linear_uni),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_linear_uni)),
        "model": linear_model_uni,
        "is_univariate": True,
        "scaled": False
    }
    
    # Evaluate SVR univariate
    y_pred_svr_uni = svr_model_uni.predict(X_test_uni_scaled)
    results["nonlinear_uni"] = {
        "r2": r2_score(y_test, y_pred_svr_uni),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_svr_uni)),
        "model": svr_model_uni,
        "is_univariate": True,
        "scaled": True,
        "scaler": scaler_uni
    }
    
    # Evaluate RF multivariate
    y_pred_rf_multi = rf_model_multi.predict(X_test)
    results["multivariate"] = {
        "r2": r2_score(y_test, y_pred_rf_multi),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf_multi)),
        "model": rf_model_multi,
        "is_univariate": False,
        "scaled": False
    }
    
    return results

# Function to evaluate models
def evaluate_models(models, X, y):
    # Prepare test data (just for displaying metrics)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Extract metrics from the models dictionary
    results = {}
    for name, model_dict in models.items():
        results[name] = {
            "r2": model_dict["r2"],
            "rmse": model_dict["rmse"]
        }
    
    return results

# Function to calculate feature importance for the multivariate model
@st.cache_resource
def calculate_feature_importance(_model, X):
    # Get feature importance from Random Forest
    importances = _model["multivariate"]["model"].feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

# Function to generate explanation text based on data insights
def generate_insights(data):
    # Calculate key insights with more details
    country_waste = data.groupby('Country')['Total Waste (Tons)'].mean().sort_values(ascending=False)
    top_waste_country = country_waste.index[0]
    top_waste_value = country_waste.values[0]
    
    country_percapita = data.groupby('Country')['Avg Waste per Capita (Kg)'].mean().sort_values(ascending=False)
    top_percapita_country = country_percapita.index[0]
    top_percapita_value = country_percapita.values[0]
    
    # Add 2nd and 3rd places
    second_waste_country = country_waste.index[1]
    second_waste_value = country_waste.values[1]
    third_waste_country = country_waste.index[2]
    third_waste_value = country_waste.values[2]
    
    second_percapita_country = country_percapita.index[1]
    second_percapita_value = country_percapita.values[1]
    
    # Food category analysis
    category_data = data.groupby('Food Category')['Total Waste (Tons)'].sum().sort_values(ascending=False)
    top_category = category_data.index[0]
    category_value = category_data.values[0]
    second_category = category_data.index[1]
    second_category_value = category_data.values[1]
    
    # Calculate year-over-year growth with more details
    yearly_waste = data.groupby('Year')['Total Waste (Tons)'].mean().reset_index()
    years = sorted(data['Year'].unique())
    
    growth_rates = []
    if len(years) > 1:
        for i in range(1, len(years)):
            current_year = years[i]
            prev_year = years[i-1]
            current_waste = yearly_waste[yearly_waste['Year'] == current_year]['Total Waste (Tons)'].values[0]
            prev_waste = yearly_waste[yearly_waste['Year'] == prev_year]['Total Waste (Tons)'].values[0]
            growth = ((current_waste - prev_waste) / prev_waste) * 100
            growth_rates.append((current_year, growth))
        
        latest_growth = growth_rates[-1][1]
        highest_growth_year, highest_growth = max(growth_rates, key=lambda x: x[1])
    else:
        latest_growth = 0
        highest_growth_year = "N/A"
        highest_growth = 0
    
    # Household waste insights
    household_pct = data['Household Waste (%)'].mean()
    
    insights = {
        "waste_by_country": {
            "top": {"country": top_waste_country, "value": top_waste_value},
            "second": {"country": second_waste_country, "value": second_waste_value},
            "third": {"country": third_waste_country, "value": third_waste_value}
        },
        "waste_per_capita": {
            "top": {"country": top_percapita_country, "value": top_percapita_value},
            "second": {"country": second_percapita_country, "value": second_percapita_value}
        },
        "food_categories": {
            "top": {"category": top_category, "value": category_value},
            "second": {"category": second_category, "value": second_category_value}
        },
        "growth": {
            "latest": latest_growth,
            "highest": {"year": highest_growth_year, "value": highest_growth}
        },
        "household_pct": household_pct
    }
    
    return insights

# Main app
def main():
    # Load data
    data = load_data()
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Sidebar options
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        ["Global Overview", "Country Analysis", "Predictive Models", "Comparative Analysis", "Insights & Explanations"]
    )
    
    # Features for modeling
    features = ["Economic Loss (Million $)", "Avg Waste per Capita (Kg)"]
    target = "Total Waste (Tons)"
    
    X = data[features]
    y = data[target]
    
    # Train models
    models = train_models(X, y)
    model_results = evaluate_models(models, X, y)
    
    # Generate feature importance
    feature_importance = calculate_feature_importance(models, X)
    
    # Global Overview
    if analysis_type == "Global Overview":
        st.markdown("<h1 class='main-header'>🌍 Global Food Waste Dashboard</h1>", unsafe_allow_html=True)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{len(data['Country'].unique()):,}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Countries</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{data['Total Waste (Tons)'].sum()/1000000:.1f}M</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Total Waste (Tons)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>${data['Economic Loss (Million $)'].sum()/1000:.1f}B</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Economic Loss</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{data['Avg Waste per Capita (Kg)'].mean():.1f}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Avg Waste per Capita (Kg)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Global map
        st.markdown("<h2 class='sub-header'>Food Waste Across the World</h2>", unsafe_allow_html=True)
        
        country_data = data.groupby('Country').agg({
            'Total Waste (Tons)': 'mean',
            'Economic Loss (Million $)': 'mean',
            'Avg Waste per Capita (Kg)': 'mean',
            'M49 code': 'first',
            'Region': 'first'
        }).reset_index()
        
        map_metric = st.selectbox("Select Map Metric", 
                                ["Total Waste (Tons)", "Economic Loss (Million $)", "Avg Waste per Capita (Kg)"])
        
        fig = px.choropleth(
            country_data,
            locations="Country",
            locationmode="country names",
            color=map_metric,
            hover_name="Country",
            hover_data=["Region", map_metric],
            title=f"Global {map_metric} by Country",
            color_continuous_scale="Viridis",
            projection="natural earth"
        )
        
        fig.update_layout(
            height=600, 
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            geo=dict(
                bgcolor='rgba(0,0,0,0)',
                lakecolor='#203354',
                landcolor='rgba(50, 50, 50, 0.2)',
                subunitcolor='gray'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top waste by region
        st.markdown("<h2 class='sub-header'>Food Waste by Region</h2>", unsafe_allow_html=True)
        
        region_data = data.groupby('Region').agg({
            'Total Waste (Tons)': 'sum',
            'Economic Loss (Million $)': 'sum'
        }).reset_index().sort_values('Total Waste (Tons)', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                region_data, 
                x='Region', 
                y='Total Waste (Tons)',
                title="Total Waste by Region (Tons)",
                color='Total Waste (Tons)',
                color_continuous_scale="Viridis",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.pie(
                region_data, 
                values='Economic Loss (Million $)', 
                names='Region',
                title="Economic Loss Distribution by Region",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis,
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Country Analysis
    elif analysis_type == "Country Analysis":
        st.markdown("<h1 class='main-header'>🌏 Country-Level Analysis</h1>", unsafe_allow_html=True)
        
        # Country selection
        country = st.selectbox("Select a Country:", sorted(data["Country"].unique()))
        
        # Filter data for selected country
        country_data = data[data["Country"] == country]
        
        # Country header with flag emoji placeholder
        st.markdown(f"<h2 class='sub-header'>Analysis for {country}</h2>", unsafe_allow_html=True)
        
        # Key metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{country_data['Total Waste (Tons)'].mean():,.0f}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Average Waste (Tons)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>${country_data['Economic Loss (Million $)'].mean():,.0f}M</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Average Economic Loss</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{country_data['Avg Waste per Capita (Kg)'].mean():.1f}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Avg Waste per Capita (Kg)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Country waste trend by year
        st.markdown("<h3 class='sub-header'>Waste Trends Over Time</h3>", unsafe_allow_html=True)
        
        yearly_data = country_data.groupby('Year').agg({
            'Total Waste (Tons)': 'mean',
            'Economic Loss (Million $)': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                yearly_data, 
                x='Year', 
                y='Total Waste (Tons)',
                markers=True,
                title=f"Waste Trend for {country} (2018-2024)",
                line_shape="spline",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.line(
                yearly_data, 
                x='Year', 
                y='Economic Loss (Million $)',
                markers=True,
                title=f"Economic Loss Trend for {country} (2018-2024)",
                line_shape="spline",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Food category breakdown
        st.markdown("<h3 class='sub-header'>Waste by Food Category</h3>", unsafe_allow_html=True)
        
        category_data = country_data.groupby('Food Category').agg({
            'Total Waste (Tons)': 'sum'
        }).reset_index().sort_values('Total Waste (Tons)', ascending=False)
        
        fig = px.bar(
            category_data,
            x='Food Category',
            y='Total Waste (Tons)',
            color='Food Category',
            title=f"Food Waste by Category in {country}",
            labels={'Total Waste (Tons)': 'Total Waste (Tons)', 'Food Category': 'Category'},
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Household vs. Retail vs. Food Service
        st.markdown("<h3 class='sub-header'>Waste Distribution by Sector</h3>", unsafe_allow_html=True)
        
        if not country_data.empty:
            household = country_data['Household estimate (tonnes/year)'].mean()
            retail = country_data['Retail estimate (tonnes/year)'].mean()
            food_service = country_data['Food service estimate (tonnes/year)'].mean()
            
            sector_data = pd.DataFrame({
                'Sector': ['Household', 'Retail', 'Food Service'],
                'Waste (Tonnes)': [household, retail, food_service]
            })
            
            fig = px.pie(
                sector_data,
                values='Waste (Tonnes)',
                names='Sector',
                title=f"Waste Distribution by Sector in {country}",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Predictive Models
    elif analysis_type == "Predictive Models":
        st.markdown("<h1 class='main-header'>🔮 Predictive Models</h1>", unsafe_allow_html=True)
        
        # Model information
        st.markdown("""
        <div class='card'>
        <h3 style='color: #1976D2;'>About the Models</h3>
        <p>We've trained three different models to predict total food waste:</p>
        <ol>
            <li><strong>Linear Regression (Univariate)</strong>: Takes a single input feature and predicts the output linearly</li>
            <li><strong>Support Vector Regression (Univariate)</strong>: A non-linear model that uses a single input feature</li>
            <li><strong>Random Forest (Multivariate)</strong>: Uses multiple input features to make predictions</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Model visualization tab
        model_tab = st.radio(
            "Select View",
            ["Performance Comparison", "Prediction Tool", "Model Visualization"]
        )
        
        if model_tab == "Performance Comparison":
            # Model performance comparison
            st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
            
            # Create a dataframe for model performance
            performance = pd.DataFrame({
                'Model': list(model_results.keys()),
                'R² Score': [model_results[m]['r2'] for m in model_results],
                'RMSE': [model_results[m]['rmse'] for m in model_results]
            })
            
            # Improve model names for display
            performance['Model'] = performance['Model'].map({
                'linear_uni': 'Linear (Univariate)',
                'nonlinear_uni': 'SVR (Univariate)',
                'multivariate': 'Random Forest (Multivariate)'
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    performance,
                    x='Model',
                    y='R² Score',
                    color='Model',
                    title="Model Performance: R² Score (higher is better)",
                    text_auto='.3f',
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.bar(
                    performance,
                    x='Model',
                    y='RMSE',
                    color='Model',
                    title="Model Performance: RMSE (lower is better)",
                    text_auto='.1f',
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Add feature importance
            st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
            
            fig = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                color='Importance',
                title="Feature Importance in the Multivariate Model",
                color_continuous_scale='Viridis',
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_tab == "Prediction Tool":
            # Model selector for predictions
            st.markdown("<h2 class='sub-header'>Make a Prediction</h2>", unsafe_allow_html=True)
            
            # Country selection for default values
            country = st.selectbox("Select a Country for Default Values:", sorted(data["Country"].unique()))
            country_data = data[data["Country"] == country]
            
            # Select which model to use
            model_choice = st.radio(
                "Select Model for Prediction",
                list(models.keys()),
                format_func=lambda x: {
                    "linear_uni": "Linear Regression (Univariate)", 
                    "nonlinear_uni": "Support Vector Regression (Univariate)", 
                    "multivariate": "Random Forest (Multivariate)"
                }[x]
            )
            
            # Get the selected model
            selected_model = models[model_choice]
            
            # Input fields based on model type
            if selected_model["is_univariate"]:
                # Univariate model requires only one input
                st.markdown("<h3>Univariate Model Input</h3>", unsafe_allow_html=True)
                feature_name = X.columns[0]  # First feature is used for univariate
                
                # Single input for univariate model
                # Determine max value based on feature type
                if "Capita" in feature_name:
                    max_val = 500.0
                else:
                    max_val = float(data[feature_name].max()) * 1.2

                # Get a safe default value
                default_val = float(country_data[feature_name].mean()) if not country_data.empty else 100.0
                default_val = min(default_val, max_val)

                # Number input
                feature_value = st.number_input(
                    f"{feature_name}", 
                    value=default_val,
                    min_value=0.0,
                    max_value=max_val,
                    step=1.0
                )

                
                # Create input for prediction
                input_data = np.array([[feature_value]])
                
                # Make prediction
                if st.button("Predict Total Waste"):
                    if selected_model["scaled"]:
                        input_scaled = selected_model["scaler"].transform(input_data)
                        prediction = selected_model["model"].predict(input_scaled)[0]
                    else:
                        prediction = selected_model["model"].predict(input_data)[0]
                        
                    st.markdown(
                        f"""
                        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32; margin-top: 20px;'>
                            <h3 style='color: #2E7D32; margin-top: 0;'>Prediction Result</h3>
                            <p>Using the <strong>{model_choice.replace('_', ' ').title()}</strong> with input <strong>{feature_name} = {feature_value}</strong>:</p>
                            <h2 style='color: #2E7D32; text-align: center;'>{prediction:,.2f} Tons</h2>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            else:
                # Multivariate model requires multiple inputs
                st.markdown("<h3>Multivariate Model Inputs</h3>", unsafe_allow_html=True)
                
                # Multiple input fields
                input_values = {}
                
                for feature in X.columns:
    # Dynamic max value
                    if "Capita" in feature:
                        max_val = 500.0
                        step_val = 1.0
                    else:
                        max_val = float(data[feature].max()) * 1.2
                        step_val = 10.0

                    # Safe default value
                    default_val = float(country_data[feature].mean()) if not country_data.empty else 100.0
                    default_val = min(default_val, max_val)

                    # Number input
                    input_values[feature] = st.number_input(
                        f"{feature}", 
                        value=default_val,
                        min_value=0.0,
                        max_value=max_val,
                        step=step_val
                    )

                
                # Create input DataFrame for prediction
                input_data = pd.DataFrame([input_values])
                
                # Make prediction
                if st.button("Predict Total Waste"):
                    prediction = selected_model["model"].predict(input_data)[0]
                    
                    st.markdown(
                        f"""
                        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32; margin-top: 20px;'>
                            <h3 style='color: #2E7D32; margin-top: 0;'>Prediction Result</h3>
                            <p>Using the <strong>Multivariate Random Forest</strong> model with these inputs:</p>
                            <ul>
                                {"".join([f"<li><strong>{feature}:</strong> {value}</li>" for feature, value in input_values.items()])}
                            </ul>
                            <h2 style='color: #2E7D32; text-align: center;'>{prediction:,.2f} Tons</h2>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
        
        elif model_tab == "Model Visualization":
            # Model visualization 
            st.markdown("<h2 class='sub-header'>Model Behavior Visualization</h2>", unsafe_allow_html=True)
            
            # Select model to visualize
            model_viz_choice = st.selectbox(
                "Select Model to Visualize",
                ["Linear (Univariate)", "Non-linear (Univariate)", "Compare Both Univariate Models"]
            )
            
            # Generate test points for visualization
            test_range = np.linspace(0, 200, 100).reshape(-1, 1)
            
            if model_viz_choice == "Linear (Univariate)" or model_viz_choice == "Compare Both Univariate Models":
                # Linear model predictions
                linear_preds = models["linear_uni"]["model"].predict(test_range)
                
                if model_viz_choice == "Linear (Univariate)":
                    fig = px.line(
                        x=test_range.flatten(), 
                        y=linear_preds,
                        labels={"x": X.columns[0], "y": "Predicted Total Waste (Tons)"},
                        title=f"Linear Model Prediction Pattern",
                        template="plotly_dark"
                    )
                    
                    # Add some data points for context
                    sample_data = data.sample(n=min(100, len(data)))
                    fig.add_scatter(
                        x=sample_data[X.columns[0]], 
                        y=sample_data[target],
                        mode='markers',
                        name='Actual Data Points',
                        marker=dict(size=8, opacity=0.6)
                    )
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation
                    st.markdown("""
                    <div class='explanation'>
                        <p>The linear model assumes a straight-line relationship between the input feature and the predicted waste. 
                        The blue line shows the model's predictions across different input values, while the scattered points show actual data.</p>
                        <p>The slope of the line indicates how much the waste increases per unit increase in the input feature.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if model_viz_choice == "Non-linear (Univariate)" or model_viz_choice == "Compare Both Univariate Models":
                # SVR model predictions
                scaled_test_range = models["nonlinear_uni"]["scaler"].transform(test_range)
                svr_preds = models["nonlinear_uni"]["model"].predict(scaled_test_range)
                
                if model_viz_choice == "Non-linear (Univariate)":
                    fig = px.line(
                        x=test_range.flatten(), 
                        y=svr_preds,
                        labels={"x": X.columns[0], "y": "Predicted Total Waste (Tons)"},
                        title=f"Non-linear (SVR) Model Prediction Pattern",
                        template="plotly_dark"
                    )
                    
                    # Add some data points for context
                    sample_data = data.sample(n=min(100, len(data)))
                    fig.add_scatter(
                        x=sample_data[X.columns[0]], 
                        y=sample_data[target],
                        mode='markers',
                        name='Actual Data Points',
                        marker=dict(size=8, opacity=0.6)
                    )
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation
                    st.markdown("""
                    <div class='explanation'>
                        <p>The SVR model can capture non-linear relationships between the input feature and waste predictions.
                        Notice how the prediction line can curve and adapt to patterns in the data that aren't simple straight lines.</p>
                        <p>This flexibility allows the model to potentially capture more complex relationships in the data.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            if model_viz_choice == "Compare Both Univariate Models":
                # Create comparison figure
                fig = go.Figure()
                
                # Add linear model line
                fig.add_trace(go.Scatter(
                    x=test_range.flatten(),
                    y=linear_preds,
                    mode='lines',
                    name='Linear Model',
                    line=dict(color='#2196F3', width=2)
                ))
                
                # Add SVR model line
                fig.add_trace(go.Scatter(
                    x=test_range.flatten(),
                    y=svr_preds,
                    mode='lines',
                    name='SVR Model (Non-linear)',
                    line=dict(color='#F44336', width=2)
                ))
                
                # Add some data points for context
                sample_data = data.sample(n=min(100, len(data)))
                fig.add_trace(go.Scatter(
                    x=sample_data[X.columns[0]],
                    y=sample_data[target],
                    mode='markers',
                    name='Actual Data Points',
                    marker=dict(size=8, opacity=0.5, color='#9E9E9E')
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"Linear vs. Non-linear Model Comparison",
                    xaxis_title=X.columns[0],
                    yaxis_title="Predicted Total Waste (Tons)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation of differences
                st.markdown("""
                <div class='explanation'>
                    <p>This comparison shows how linear and non-linear models make different predictions with the same input data:</p>
                    <ul>
                        <li>The <strong style="color:blue">blue line</strong> (linear model) can only fit a straight line through the data.</li>
                        <li>The <strong style="color:red">red line</strong> (non-linear SVR model) can adapt to more complex patterns.</li>
                    </ul>
                    <p>Notice where the models differ most - these represent areas where non-linear relationships in the data become important.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Comparative Analysis
    elif analysis_type == "Comparative Analysis":
        st.markdown("<h1 class='main-header'>📊 Comparative Analysis</h1>", unsafe_allow_html=True)
        
        # Country comparison
        st.markdown("<h2 class='sub-header'>Compare Countries</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            countries = st.multiselect(
                "Select Countries to Compare:", 
                sorted(data["Country"].unique()),
                default=sorted(data["Country"].unique())[:5]
            )
            
        with col2:
            metric = st.selectbox(
                "Select Comparison Metric:",
                ["Total Waste (Tons)", "Economic Loss (Million $)", "Avg Waste per Capita (Kg)"]
            )
        
        if countries:
            # Filter data for selected countries
            filtered_data = data[data["Country"].isin(countries)]
            
            # Group by country
            country_comparison = filtered_data.groupby('Country').agg({
                metric: 'mean'
            }).reset_index().sort_values(metric, ascending=False)
            
            # Create comparison chart
            fig = px.bar(
                country_comparison,
                x='Country',
                y=metric,
                color='Country',
                title=f"Comparison of {metric} by Country",
                text_auto='.1f',
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Regional comparison
        st.markdown("<h2 class='sub-header'>Compare Regions</h2>", unsafe_allow_html=True)
        
        regions = st.multiselect(
            "Select Regions to Compare:",
            sorted(data["Region"].unique()),
            default=sorted(data["Region"].unique())[:3]
        )
        
        if regions:
            # Filter data for selected regions
            filtered_data = data[data["Region"].isin(regions)]
            
            # Group by region and year
            region_year_data = filtered_data.groupby(['Region', 'Year']).agg({
                'Total Waste (Tons)': 'mean'
            }).reset_index()
            
            # Create line chart for trends
            fig = px.line(
                region_year_data,
                x='Year',
                y='Total Waste (Tons)',
                color='Region',
                title="Waste Trends by Region Over Time",
                markers=True,
                line_shape="spline",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Food category comparison
        st.markdown("<h2 class='sub-header'>Waste by Food Category</h2>", unsafe_allow_html=True)
        
        # Group by food category
        category_data = data.groupby('Food Category').agg({
            'Total Waste (Tons)': 'sum',
            'Economic Loss (Million $)': 'sum'
        }).reset_index().sort_values('Total Waste (Tons)', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                category_data,
                x='Food Category',
                y='Total Waste (Tons)',
                color='Food Category',
                title="Total Waste by Food Category",
                text_auto='.1f',
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.pie(
                category_data,
                values='Economic Loss (Million $)',
                names='Food Category',
                title="Economic Loss by Food Category",
                hole=0.4,
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Correlation analysis
        st.markdown("<h2 class='sub-header'>Correlation Analysis</h2>", unsafe_allow_html=True)
        
        # Get numeric columns for correlation analysis
        numeric_cols = ['Total Waste (Tons)', 'Economic Loss (Million $)', 'Avg Waste per Capita (Kg)', 
                        'Population (Million)', 'Household Waste (%)']
        
        corr_data = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_data,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of Key Metrics",
            zmin=-1, zmax=1,
            template="plotly_dark"
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Insights & Explanations
    elif analysis_type == "Insights & Explanations":
        st.markdown("<h1 class='main-header'>🧠 Key Insights & Explanations</h1>", unsafe_allow_html=True)
        
        # Generate key insights with the improved function
        insights_data = generate_insights(data)
        
        # Display key insights with better formatting
        st.markdown("<h2 class='sub-header'>📊 Key Data Insights</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Top Countries by Total Waste</div>
                <div class="insight-value">1. {insights_data["waste_by_country"]["top"]["country"]}</div>
                <p>{insights_data["waste_by_country"]["top"]["value"]:,.0f} tons on average</p>
                <div class="insight-value">2. {insights_data["waste_by_country"]["second"]["country"]}</div>
                <p>{insights_data["waste_by_country"]["second"]["value"]:,.0f} tons on average</p>
                <div class="insight-value">3. {insights_data["waste_by_country"]["third"]["country"]}</div>
                <p>{insights_data["waste_by_country"]["third"]["value"]:,.0f} tons on average</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Growth Rate</div>
                <p>Latest year-over-year waste growth rate: <span class="insight-value">{insights_data["growth"]["latest"]:.1f}%</span></p>
                <p>Highest growth occurred in {insights_data["growth"]["highest"]["year"]} at <span class="insight-value">{insights_data["growth"]["highest"]["value"]:.1f}%</span></p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Top Countries by Per Capita Waste</div>
                <div class="insight-value">1. {insights_data["waste_per_capita"]["top"]["country"]}</div>
                <p>{insights_data["waste_per_capita"]["top"]["value"]:.1f} kg per person</p>
                <div class="insight-value">2. {insights_data["waste_per_capita"]["second"]["country"]}</div>
                <p>{insights_data["waste_per_capita"]["second"]["value"]:.1f} kg per person</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Top Wasted Food Categories</div>
                <div class="insight-value">1. {insights_data["food_categories"]["top"]["category"]}</div>
                <p>{insights_data["food_categories"]["top"]["value"]:,.0f} tons globally</p>
                <div class="insight-value">2. {insights_data["food_categories"]["second"]["category"]}</div>
                <p>{insights_data["food_categories"]["second"]["value"]:,.0f} tons globally</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add a pie chart showing distribution of waste by food category
        st.markdown("<h3 class='sub-header'>Food Category Distribution</h3>", unsafe_allow_html=True)
        
        category_data = data.groupby('Food Category')['Total Waste (Tons)'].sum().reset_index()
        
        fig = px.pie(
            category_data,
            values='Total Waste (Tons)',
            names='Food Category',
            title="Distribution of Food Waste by Category",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4,
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model explanation section with improved styling
        st.markdown("<h2 class='sub-header'>🧪 Model Explanations</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='explanation'>
            <h3>Linear Regression (Univariate Model)</h3>
            <p>This model uses a single input feature to predict food waste using a straight-line relationship:</p>
            <ul>
                <li><strong>Formula</strong>: Total Waste = (Coefficient × Input Feature) + Intercept</li>
                <li><strong>Strength</strong>: Simple, interpretable, and works well for linear relationships</li>
                <li><strong>Limitation</strong>: Cannot capture complex non-linear patterns in the data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='explanation'>
            <h3>Support Vector Regression (Non-linear Univariate Model)</h3>
            <p>This model uses a single input feature but can capture non-linear relationships:</p>
            <ul>
                <li><strong>Technique</strong>: Uses kernel functions to transform the data into a higher-dimensional space</li>
                <li><strong>Strength</strong>: Can model complex patterns that aren't simple straight lines</li>
                <li><strong>Limitation</strong>: Less interpretable than linear models; requires careful tuning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='explanation'>
            <h3>Random Forest (Multivariate Model)</h3>
            <p>This model uses multiple input features to make predictions:</p>
            <ul>
                <li><strong>Technique</strong>: Builds many decision trees and averages their predictions</li>
                <li><strong>Strength</strong>: Can capture complex interactions between multiple features</li>
                <li><strong>Limitation</strong>: More complex and computationally intensive</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance visualization with improved styling
        st.markdown("<h2 class='sub-header'>📈 Feature Importance Analysis</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-importance-card'>
            <h3>What Drives Food Waste?</h3>
            <p>Feature importance shows which input variables have the most influence on predictions. 
            This helps identify the key factors driving food waste.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a more colorful and clear feature importance chart
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance in Predicting Food Waste',
            color='Importance',
            color_continuous_scale='Viridis',  # Changed color scale
            template='plotly_dark'  # Using dark template
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Relative Importance (%)",
            yaxis_title="Feature",
            font=dict(size=14),
            title_font=dict(size=20),
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            coloraxis_colorbar=dict(
                title="Importance",
            )
        )
        
        # Convert the importance to percentages for easier understanding
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.1%}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add feature importance interpretation 
        st.markdown("""
        <div class='feature-importance-card'>
            <h3>Interpretation</h3>
            <p>The chart above shows that <strong>Economic Loss</strong> and <strong>Average Waste per Capita</strong> 
            are the key factors influencing total food waste predictions. This suggests that addressing economic factors 
            and per-capita consumption patterns would be most effective in reducing overall food waste.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations section with improved styling
        st.markdown("<h2 class='sub-header'>🚀 Recommendations</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='recommendation-card'>
            <h3>Policy Recommendations</h3>
            <p>Based on the data analysis and model insights, here are potential strategies to reduce food waste:</p>
            <ol>
                <li><strong>Target high waste-per-capita regions</strong> with consumer education campaigns focusing on Germany and other high per-capita waste countries</li>
                <li><strong>Focus on Prepared Food and other top waste categories</strong> in retail and distribution systems</li>
                <li><strong>Address economic factors</strong> that show strong correlation with waste generation through incentives and penalties</li>
                <li><strong>Implement household waste reduction programs</strong> as this sector contributes significantly to the overall waste</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Add an actionable insights section
        st.markdown("""
        <div class='recommendation-card'>
            <h3>Actionable Insights for Stakeholders</h3>
            <ul>
                <li><strong>Food Retailers:</strong> Implement inventory management systems that predict demand more accurately to reduce overstocking</li>
                <li><strong>Policymakers:</strong> Develop tax incentives for businesses that donate excess food rather than discard it</li>
                <li><strong>Consumers:</strong> Plan meals better, understand food date labels, and learn proper food storage techniques</li>
                <li><strong>Researchers:</strong> Focus on developing better preservation technologies for the most wasted food categories</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Food Waste Analytics Dashboard - CS316 Project</p>
        <p>Data source: UN Food Waste Index | Dashboard Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
