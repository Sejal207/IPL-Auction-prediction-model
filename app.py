import streamlit as st

st.set_page_config(
    page_title="IPL Auction Predictor",
    page_icon="🏏",
    layout="wide"
)

st.title("IPL Player Auction Price Prediction")

st.image("https://www.mpl.live/blog/wp-content/uploads/2021/03/ipl-logo-new-old-1200x900.jpg", width=200)

st.markdown("""
The Indian Premier League (IPL), since its inception in 2008, has grown to become one of the most popular and financially successful cricket leagues in the world. What sets the IPL apart is its auction system, where franchise owners bid for players, creating a dynamic market where a cricketer's value is determined not only by their on-field performance but also by a variety of other factors. This auction process, while exciting, also poses a challenge: how can franchise owners accurately predict the value of a player and make informed decisions on how much to bid?

Machine learning (ML) provides a powerful solution to this challenge by leveraging data-driven models to predict player auction prices. By analyzing historical player performance data, team strategies, and other influencing factors, ML models can help identify patterns and relationships that affect a player's value at auction. These models can process vast amounts of data, including player statistics such as batting averages, strike rates, wickets, and economy rates, as well as off-field factors like a player’s popularity and marketability, which may significantly influence their price.

In this study, we focus on predicting the winning bid for IPL players in upcoming auctions using a variety of machine learning techniques. We apply models like Linear Regression, Lasso Regression, Ridge Regression, Random Forest, and XGBoost to predict player prices based on key features extracted from their career data. The model incorporates feature engineering, exploratory data analysis (EDA), data cleaning, and web scraping techniques to collect and prepare the data. Additionally, methods like outlier detection, hyperparameter tuning, and Principal Component Analysis (PCA)-optimized XGBoost are used to enhance the model's performance and accuracy.

Key features influencing the auction price include performance statistics such as batting and bowling averages, strike rates, and number of wickets, as well as contextual factors like a player’s previous IPL performances. By utilizing these features, the model aims to predict auction prices that are as close to the actual bid as possible.

This research not only contributes to the field of sports analytics but also offers valuable insights to IPL franchise owners, enabling them to make better, data-driven decisions on player acquisitions. With the ever-growing importance of big data and machine learning in sports, this approach has the potential to revolutionize the IPL auction process, leading to more strategic, informed, and cost-effective decisions.
""")

st.sidebar.success("Select a page above to get started.")

