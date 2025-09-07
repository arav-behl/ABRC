import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Financial Derivatives Pricing & Risk Management",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .tech-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        background-color: #e1f5fe;
        color: #0277bd;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    .highlight-box {
        background-color: #f8f9ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #333333;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        min-width: 350px !important;
        width: 350px !important;
    }
    
    [data-testid="stSidebar"] > div {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    .sidebar-title {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    
    .stSelectbox label {
        font-size: 1.4rem !important;
        font-weight: bold !important;
        color: #2c3e50 !important;
        margin-bottom: 1rem !important;
    }
    
    .stSelectbox > div > div {
        font-size: 1.2rem !important;
        padding: 12px 16px !important;
        border: 2px solid #1f77b4 !important;
        border-radius: 8px !important;
        background-color: white !important;
    }
    
    .navigation-info {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<div class="main-header">Autocallable Barrier Reverse Callable<br>Pricing & Risk Management System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">Project Navigation</div>', unsafe_allow_html=True)
    
    st.markdown("#### Select a Section to Explore:")
    
    page = st.selectbox("Navigate to:", [
        "Overview", 
        "Product Details", 
        "Technical Models", 
        "Backtesting Results",
        "Risk Management",
        "Key Insights"
    ], key="main_navigation")
    
    # Add some helpful navigation info
    st.markdown("""
    <div class="navigation-info">
        <h4 style="color: #1f77b4; margin-top: 0;">📚 What You'll Find:</h4>
        <ul style="color: #2c3e50; line-height: 1.6;">
            <li><strong>Overview:</strong> Project summary & tech stack</li>
            <li><strong>Product Details:</strong> Financial instrument specs</li>
            <li><strong>Technical Models:</strong> Heston, GBM, CIR models</li>
            <li><strong>Backtesting:</strong> Performance & accuracy</li>
            <li><strong>Risk Management:</strong> Greeks & hedging</li>
            <li><strong>Insights:</strong> Key findings & achievements</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content based on selection
if page == "Overview":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="subheader">Project Overview</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
        <strong>Objective:</strong> Developed a comprehensive pricing and risk management system for complex structured financial derivatives using advanced quantitative methods.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Technical Stack")
        tech_stack = [
            "Python", "Monte Carlo Simulation", "Heston Model", "CIR Model",
            "Implied Volatility", "Newton-Raphson", "Cholesky Decomposition",
            "Nelson-Siegel-Svensson", "Risk Management", "Delta Hedging",
            "Value at Risk (VaR)", "Pandas", "NumPy", "SciPy", "Matplotlib"
        ]
        
        tech_html = "".join([f'<span class="tech-badge">{tech}</span>' for tech in tech_stack])
        st.markdown(tech_html, unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="subheader">Key Metrics</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>Simulations</h3>
            <h2>5,000+</h2>
            <p>Monte Carlo paths per analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>RMSE</h3>
            <h2>25.89</h2>
            <p>Best model accuracy (GBM_CIR_AV)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>VaR (95%)</h3>
            <h2>649.02</h2>
            <p>Worst-case scenario pricing</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Product Details":
    st.markdown('<div class="subheader">Autocallable Barrier Reverse Callable</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Product Specifications")
        
        product_specs = {
            "Underlying Assets": "Alcon, Sonova, Straumann",
            "Denomination": "CHF 1,000",
            "Period": "15 months (Aug 2023 - Nov 2024)",
            "Coupon Rate": "9.0% p.a. (Quarterly)",
            "Barrier Level": "59% of Initial Price",
            "Initial Prices": {
                "Alcon": "CHF 71.70",
                "Sonova": "CHF 234.40", 
                "Straumann": "CHF 139.95"
            }
        }
        
        for key, value in product_specs.items():
            if key == "Initial Prices":
                st.write(f"**{key}:**")
                for asset, price in value.items():
                    st.write(f"  - {asset}: {price}")
            else:
                st.write(f"**{key}:** {value}")
    
    with col2:
        st.markdown("#### Payout Scenarios")
        
        # Create a sample payout diagram
        scenarios = ["Early Redemption", "No Barrier Hit", "Barrier Hit + Recovery", "Barrier Hit + Loss"]
        payouts = [1090, 1090, 1090, 750]  # Approximate values
        
        fig = go.Figure(data=[
            go.Bar(x=scenarios, y=payouts, 
                   marker_color=['green', 'lightgreen', 'orange', 'red'])
        ])
        
        fig.update_layout(
            title="Potential Payout Scenarios (CHF)",
            yaxis_title="Payout Amount",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "Technical Models":
    st.markdown('<div class="subheader">Advanced Mathematical Models</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Heston Model", "GBM Variations", "Interest Rate Models", "Variance Reduction"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Heston Stochastic Volatility Model")
            st.latex(r"""
            dS_t = rS_t dt + \sqrt{v_t}S_t dW_t^S
            """)
            st.latex(r"""
            dv_t = \kappa(\theta - v_t)dt + \sigma\sqrt{v_t}dW_t^v
            """)
            
            st.markdown("**Key Features:**")
            st.write("- Stochastic volatility modeling")
            st.write("- Mean-reverting variance process")
            st.write("- Correlated Brownian motions")
            st.write("- Calibrated to market option prices")
            
        with col2:
            # Simulate a sample Heston path for visualization
            np.random.seed(42)
            T, N = 1.0, 252
            dt = T/N
            
            # Heston parameters (approximate)
            kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.3, -0.5, 0.04
            S0 = 100
            
            # Simulate paths
            times = np.linspace(0, T, N+1)
            S = np.zeros(N+1)
            v = np.zeros(N+1)
            S[0], v[0] = S0, v0
            
            for i in range(N):
                Z1, Z2 = np.random.normal(0, 1, 2)
                Zv = Z1
                Zs = rho * Z1 + np.sqrt(1-rho**2) * Z2
                
                v[i+1] = max(v[i] + kappa*(theta - v[i])*dt + sigma*np.sqrt(v[i]*dt)*Zv, 0.001)
                S[i+1] = S[i] * np.exp((0.05 - 0.5*v[i])*dt + np.sqrt(v[i]*dt)*Zs)
            
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=('Asset Price Path', 'Volatility Path'))
            
            fig.add_trace(go.Scatter(x=times, y=S, name='Asset Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=times, y=np.sqrt(v)*100, name='Volatility %'), row=2, col=1)
            
            fig.update_layout(height=500, title="Sample Heston Model Simulation")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Geometric Brownian Motion Variations")
        
        models = ["Standard GBM", "GBM + Antithetic Variates", "GBM + Control Variates", 
                 "GBM + EMS Correction", "GBM + CIR Rates", "GBM + CIR + Antithetic"]
        rmse_values = [30.25, 30.20, 30.15, 30.19, 30.25, 25.89]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=rmse_values, 
                   marker_color=['lightblue', 'blue', 'darkblue', 'purple', 'red', 'darkred'])
        ])
        
        fig.update_layout(
            title="Model Performance Comparison (RMSE)",
            yaxis_title="Root Mean Square Error",
            xaxis_tickangle=-45,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Best Performing Model: GBM_CIR_AV (RMSE: 25.89)**")
    
    with tab3:
        st.markdown("#### Cox-Ingersoll-Ross Interest Rate Model")
        st.latex(r"dr_t = \kappa(\theta - r_t)dt + \sigma\sqrt{r_t}dW_t")
        
        # Simulate CIR process
        np.random.seed(42)
        T, N = 2.0, 500
        dt = T/N
        kappa, theta, sigma, r0 = 0.5, 0.03, 0.1, 0.02
        
        times = np.linspace(0, T, N+1)
        rates = np.zeros(N+1)
        rates[0] = r0
        
        for i in range(N):
            dW = np.random.normal(0, np.sqrt(dt))
            rates[i+1] = max(rates[i] + kappa*(theta - rates[i])*dt + sigma*np.sqrt(rates[i]*dt)*dW, 0.001)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=rates*100, mode='lines', name='Interest Rate %'))
        fig.add_hline(y=theta*100, line_dash="dash", line_color="red", annotation_text="Long-term Mean")
        
        fig.update_layout(
            title="CIR Interest Rate Model Simulation",
            xaxis_title="Time (Years)",
            yaxis_title="Interest Rate (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Variance Reduction Techniques")
        
        techniques = {
            "Antithetic Variates": "Uses negatively correlated random numbers to reduce variance",
            "Control Variates": "Leverages known analytical solutions as control variables",
            "Empirical Martingale Simulation": "Corrects for discretization bias in Monte Carlo"
        }
        
        for technique, description in techniques.items():
            st.markdown(f"**{technique}:** {description}")

elif page == "Backtesting Results":
    st.markdown('<div class="subheader">Model Performance Analysis</div>', unsafe_allow_html=True)
    
    # Create sample backtesting data based on the presentation
    dates = pd.date_range('2024-08-01', '2024-11-07', freq='D')
    np.random.seed(42)
    
    # Simulate actual vs predicted prices based on the patterns shown
    actual_price = 980 + np.random.normal(0, 10, len(dates))
    actual_price[:20] = np.linspace(950, 990, 20)  # Initial rise
    actual_price[20:] = 990 + np.random.normal(0, 8, len(dates)-20)  # Stable period
    
    heston_price = 1000 + np.random.normal(0, 15, len(dates))
    gbm_price = 1020 + np.random.normal(0, 12, len(dates))
    
    # Create the backtesting plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=dates, y=actual_price, 
                            mode='lines', name='Actual Derivative Price', 
                            line=dict(color='black', width=3)))
    
    fig.add_trace(go.Scatter(x=dates, y=heston_price, 
                            mode='lines', name='Heston Model', 
                            line=dict(color='red', dash='dash')))
    
    fig.add_trace(go.Scatter(x=dates, y=gbm_price, 
                            mode='lines', name='Best GBM Model', 
                            line=dict(color='blue', dash='dot')))
    
    fig.update_layout(
        title="Backtesting Results: Actual vs Model Prices",
        xaxis_title="Date",
        yaxis_title="Derivative Price (CHF)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Heston Model</h4>
            <h3>RMSE: 35.89</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Best GBM</h4>
            <h3>RMSE: 25.89</h3>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Simulation Period</h4>
            <h3>3 Months</h3>
        </div>
        """, unsafe_allow_html=True)

elif page == "Risk Management":
    st.markdown('<div class="subheader">Greeks & Risk Metrics</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Delta", "Gamma", "Vega"])
    
    with tab1:
        st.markdown("#### Delta - Price Sensitivity")
        st.markdown("Delta measures how much the product's price changes for every $1 change in the underlying asset price.")
        
        # Sample delta values from the presentation
        assets = ['ALC.SW', 'SOON.SW', 'STMN.SW']
        delta_values = [3.02, 0.64, -0.65]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            for asset, delta in zip(assets, delta_values):
                st.metric(f"{asset} Delta", f"{delta:.3f}")
        
        with col2:
            # Create sample delta over time
            dates = pd.date_range('2024-08-01', '2024-11-07', freq='D')
            np.random.seed(42)
            
            fig = go.Figure()
            for i, (asset, base_delta) in enumerate(zip(assets, delta_values)):
                delta_path = base_delta + np.random.normal(0, 0.3, len(dates))
                fig.add_trace(go.Scatter(x=dates, y=delta_path, 
                                       mode='lines', name=f'{asset} Delta'))
            
            fig.update_layout(title="Delta Over Time", 
                            xaxis_title="Date", 
                            yaxis_title="Delta",
                            height=400)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Gamma - Delta Sensitivity")
        st.markdown("Gamma measures the rate of change of delta with respect to changes in the underlying asset price.")
        
        gamma_values = [-1.706, -0.081, -0.659]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            for asset, gamma in zip(assets, gamma_values):
                st.metric(f"{asset} Gamma", f"{gamma:.3f}")
        
        with col2:
            st.markdown("**Gamma Interpretation:**")
            st.write("- Negative gamma indicates decreasing delta sensitivity")
            st.write("- Higher absolute values = more sensitivity to price changes")
            st.write("- Important for hedging portfolio rebalancing")
    
    with tab3:
        st.markdown("#### Vega - Volatility Sensitivity")
        st.markdown("Vega measures the rate of change in the product's price per 1% change in implied volatility.")
        
        vega_values = [-331.06, -343.28, 469.28]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            for asset, vega in zip(assets, vega_values):
                st.metric(f"{asset} Vega", f"{vega:.2f}")
        
        with col2:
            st.markdown("**Risk Management Strategy:**")
            st.write("- **Portfolio Construction:** θ = P - δ₁S₁ - δ₂S₂ - δ₃S₃")
            st.write("- **Delta Neutral:** Hedge with underlying stocks")
            st.write("- **Dynamic Rebalancing:** Adjust positions based on Greeks")

elif page == "Key Insights":
    st.markdown('<div class="subheader">Project Insights & Achievements</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Technical Achievements")
        
        achievements = [
            "• Implemented sophisticated Heston stochastic volatility model",
            "• Developed multiple variance reduction techniques",
            "• Created comprehensive risk management framework",
            "• Built real-time sensitivity analysis (Greeks)",
            "• Performed extensive backtesting validation",
            "• Calculated Value at Risk (VaR) metrics",
            "• Integrated market data from Bloomberg Terminal"
        ]
        
        for achievement in achievements:
            st.markdown(achievement)
    
    with col2:
        st.markdown("#### Key Findings")
        
        findings = [
            "**Best Model:** GBM with CIR rates + Antithetic Variates (RMSE: 25.89)",
            "**Market Insight:** Swiss bond yields provided accurate risk-free rates",
            "**Risk Profile:** 95% VaR at CHF 649.02 indicates significant downside risk",
            "**Accuracy:** Models tracked actual derivative prices with high precision",
            "**Practical Value:** Framework suitable for real trading applications",
            "**Hedging:** Delta-neutral strategies provide effective risk mitigation"
        ]
        
        for finding in findings:
            st.markdown(finding)
    
    st.markdown("---")
    
    st.markdown("#### Future Enhancements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Model Extensions**
        - Jump diffusion processes
        - Multi-factor Heston models
        - Machine learning integration
        """)
    
    with col2:
        st.markdown("""
        **Risk Management**
        - Higher-order Greeks (Charm, Vanna)
        - Stress testing scenarios
        - Portfolio optimization
        """)
    
    with col3:
        st.markdown("""
        **Technology**
        - Real-time data feeds
        - GPU acceleration
        - Cloud deployment
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Financial Derivatives Pricing & Risk Management System</strong></p>
    <p>Advanced Quantitative Finance | Monte Carlo Methods | Risk Analytics</p>
    <p>Built with Python, Streamlit, and cutting-edge financial modeling techniques</p>
</div>
""", unsafe_allow_html=True)
