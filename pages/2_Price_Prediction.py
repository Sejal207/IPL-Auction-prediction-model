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
        return df
    except FileNotFoundError:
        st.error(f"Player data not found at {data_path}.")
        return None

# NOTE: You need to train a model and save it as 'ipl_auction_model.pkl'
# For now, we'll create a placeholder.
# Replace this with your actual model loading.
# Example: model = load_model('ipl_auction_model.pkl')
model = None  # Placeholder

player_stats = load_player_data('ipl_auction_data_2013_2025.csv') # Using auction data as a stand-in for player stats

if model is None:
    st.warning("Machine learning model not found. Price prediction is disabled.")
    st.info("To enable predictions, please train a model and save it as `ipl_auction_model.pkl`.")

if player_stats is not None and model is not None:
    # --- Player Selection ---
    st.sidebar.header("Player Selection")
    
    # Get unique player names
    player_names = sorted(player_stats['Player'].unique())
    selected_player = st.sidebar.selectbox("Select Player", player_names)

    # Get stats for the selected player
    player_data = player_stats[player_stats['Player'] == selected_player].iloc[-1] # Get latest stats

    st.header(f"Predicting Price for: {selected_player}")

    # --- Display Player Stats and Get Input ---
    st.subheader("Enter Player's Latest Stats")
    
    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)

    # Example features - you will need to adjust these to match your model's features
    with col1:
        runs = st.number_input("Runs Scored (last season)", value=int(player_data.get('Runs', 0)))
        avg = st.number_input("Batting Average (last season)", value=float(player_data.get('Avg', 0.0)))
        sr = st.number_input("Strike Rate (last season)", value=float(player_data.get('SR', 0.0)))

    with col2:
        wickets = st.number_input("Wickets Taken (last season)", value=int(player_data.get('Wkts', 0)))
        econ = st.number_input("Economy Rate (last season)", value=float(player_data.get('Econ', 0.0)))
        age = st.number_input("Player Age", value=28) # Example value

    # --- Prediction ---
    if st.button("Predict Price"):
        # Create a DataFrame from the input for the model
        # IMPORTANT: The feature names here MUST match the feature names your model was trained on
        features = pd.DataFrame({
            'Runs': [runs],
            'Batting_Avg': [avg],
            'Strike_Rate': [sr],
            'Wickets': [wickets],
            'Economy': [econ],
            'Age': [age]
            # Add other features your model expects
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
