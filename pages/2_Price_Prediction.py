import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Price Prediction", page_icon="🔮")

st.markdown("# Predict Player Auction Price")
st.sidebar.header("Predict Price")
st.write(
    """Select a player and their stats to predict their upcoming auction price."""
)

# --- Load Model and Data ---
@st.cache_resource
def load_model(model_path):
    """Loads the trained machine learning model."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None

@st.cache_data
def load_player_data(data_path):
    """Loads the player stats data."""
    try:
        df = pd.read_csv(data_path)
        # Clean player names to avoid sorting errors
        df.dropna(subset=['Player'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Player data not found at {data_path}.")
        return None

# Load the actual model
model_data = load_model('ipl_auction_model.pkl')
model = model_data['model'] if model_data else None
le_overseas = model_data['le_overseas'] if model_data else None

player_stats = load_player_data('ipl_auction_data_2013_2025.csv') # Using auction data as a stand-in for player stats

if model is None:
    st.warning("Machine learning model not found. Price prediction is disabled.")
    st.info("To enable predictions, please train a model and save it as `ipl_auction_model.pkl`.")

if player_stats is not None and model is not None:
    # --- Player Selection ---
    st.sidebar.header("Player Selection")
    
    # Get unique player names (ensuring they are strings)
    player_names = sorted(player_stats['Player'].astype(str).unique())
    selected_player = st.sidebar.selectbox("Select Player", player_names)

    # Get stats for the selected player
    player_data = player_stats[player_stats['Player'] == selected_player].iloc[-1] # Get latest stats

    st.header(f"Predicting Price for: {selected_player}")

    # --- Display Player Stats and Get Input ---
    st.subheader("Enter Player's Latest Stats")
    
    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)

    # Helper to clean currency values for default inputs
    def clean_price(price):
        if isinstance(price, (int, float)):
            return float(price)
        if isinstance(price, str):
            price = price.replace('₹', '').replace('$', '').replace(',', '').strip()
        try:
            return float(price)
        except:
            return 0.0

    with col1:
        base_price = st.number_input("Base Price (in ₹)", value=clean_price(player_data.get('Base Price', 0)))
        overseas = st.selectbox("Is Overseas Player?", options=["No", "Yes"], index=1 if player_data.get('Overseas') == 'Yes' else 0)
        runs = st.number_input("Runs Scored (last season)", value=int(player_data.get('Runs', 0)))

    with col2:
        avg = st.number_input("Batting Average (last season)", value=float(player_data.get('Avg', 0.0)))
        sr = st.number_input("Strike Rate (last season)", value=float(player_data.get('SR', 0.0)))
        wickets = st.number_input("Wickets Taken (last season)", value=int(player_data.get('Wkts', 0)))

    # --- Prediction ---
    if st.button("Predict Price"):
        # Encode Overseas
        overseas_encoded = le_overseas.transform([overseas])[0]

        # Create a DataFrame from the input for the model
        features = pd.DataFrame({
            'Base Price': [base_price],
            'Overseas_Encoded': [overseas_encoded]
        })

        try:
            # Make a prediction
            prediction = model.predict(features)[0]
            
            st.success(f"Predicted Auction Price: ₹{prediction:,.0f}")

            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please ensure the input features match what the model was trained on.")

else:
    if player_stats is None:
        st.error("Could not load player data for predictions.")
