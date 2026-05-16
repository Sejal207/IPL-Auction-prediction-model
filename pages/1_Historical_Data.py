import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Historical Data", page_icon="📊")

st.markdown("# Historical Auction Data")
st.sidebar.header("Historical Data")
st.write(
    """This page displays historical IPL auction data. Use the filters to explore the data for different years and teams."""
)

# --- Data Loading ---
@st.cache_data
def load_data(filepath):
    """Loads and cleans the IPL auction data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        
        # Helper to clean currency values
        def clean_price(price):
            if isinstance(price, str):
                price = price.replace('₹', '').replace(',', '')
            return pd.to_numeric(price, errors='coerce')

        # Clean the price columns
        for col in ['Base Price', 'Winning Bid']:
            if col in df.columns:
                df[col] = df[col].apply(clean_price)
        
        # Drop rows where winning bid is not a number, as they can't be analyzed
        df.dropna(subset=['Winning Bid'], inplace=True)
        
        return df
    except FileNotFoundError:
        st.error(f"Error: The file was not found at {filepath}")
        st.info("Please make sure the data file is in the correct directory.")
        return None

# Load the dataset
DATA_FILE = 'ipl_auction_data_2013_2025.csv'
data = load_data(DATA_FILE)

if data is not None:
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    years = sorted(data['Year'].dropna().unique(), reverse=True)
    selected_year = st.sidebar.selectbox("Select Year", years)

    filtered_data = data[data['Year'] == selected_year]

    teams = sorted(filtered_data['Team'].dropna().astype(str).unique())
    selected_teams = st.sidebar.multiselect("Select Teams", teams, default=teams)

    filtered_data = filtered_data[filtered_data['Team'].isin(selected_teams)]

    # --- Main Page Content ---
    st.header(f"Auction Data for {selected_year}")

    # --- Key Metrics ---
    total_spent = filtered_data['Winning Bid'].sum()
    num_players = len(filtered_data)
    avg_price = filtered_data['Winning Bid'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spent", f"₹{total_spent:,.0f}")
    col2.metric("Number of Players", num_players)
    col3.metric("Average Player Price", f"₹{avg_price:,.0f}")

    st.markdown("---")

    # --- Data Table ---
    st.header("Player Auction Details")
    st.dataframe(filtered_data.style.format({'Base Price': '₹{:,.0f}', 'Winning Bid': '₹{:,.0f}'}))

    # --- Visualizations ---
    st.header("Visualizations")

    # Bar chart for spending by team
    team_spending = filtered_data.groupby('Team')['Winning Bid'].sum().sort_values(ascending=False)
    fig_team_spending = px.bar(
        team_spending,
        x=team_spending.index,
        y='Winning Bid',
        title=f"Total Spending by Team in {selected_year}",
        labels={'Winning Bid': 'Total Spent (in ₹)', 'Team': 'Team'},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig_team_spending, use_container_width=True)

    # Pie chart for player roles
    if 'Role' in filtered_data.columns:
        role_distribution = filtered_data['Role'].value_counts()
        fig_role_distribution = px.pie(
            role_distribution,
            values=role_distribution.values,
            names=role_distribution.index,
            title=f"Distribution of Player Roles in {selected_year}",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_role_distribution, use_container_width=True)
    else:
        st.warning("The 'Role' column is not available in the data for the selected year.")

else:
    st.warning("Data could not be loaded. Please check the file path and try again.")
