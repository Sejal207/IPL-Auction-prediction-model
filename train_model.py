"""
IPL Auction Price Prediction - Full Training Pipeline
======================================================
This script builds a proper stacking regressor model using all available
auction data (2013-2025). Since external player stats are unavailable,
we engineer meaningful features entirely from the auction history:
  - Player reputation score (based on peak & average historical bids)
  - Bid growth trend (are they getting more expensive over time?)
  - Capped/Uncapped status, Overseas flag, RTM usage
  - Base price (in crores, normalised)
  - Auction year
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1.  LOAD & CLEAN DATA
# ──────────────────────────────────────────────

def clean_price(val):
    """Convert currency strings like ₹2,00,00,000 or $1,234 to float."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace('₹', '').replace('$', '').replace(',', '').strip()
    try:
        return float(s)
    except:
        return np.nan

def load_all_data():
    df = pd.read_csv('ipl_auction_data_2013_2025.csv')
    df['Base Price'] = df['Base Price'].apply(clean_price)
    df['Winning Bid'] = df['Winning Bid'].apply(clean_price)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Drop rows with missing critical values
    df = df.dropna(subset=['Player', 'Winning Bid', 'Year'])
    df = df[df['Winning Bid'] > 0]
    df['Player'] = df['Player'].astype(str).str.strip()
    return df

# ──────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ──────────────────────────────────────────────

def engineer_features(df):
    """Build rich features from auction history alone."""
    df = df.copy().sort_values('Year')

    # --- Encode categoricals ---
    df['Overseas_enc']  = (df['Overseas'].astype(str).str.strip().str.upper() == 'YES').astype(int)
    df['RTM_enc']       = (df['RTM'].astype(str).str.strip().str.upper() == 'YES').astype(int)

    capped_map = {'capped': 2, 'uncapped': 0}
    df['Capped_enc'] = df['Capped/Uncapped'].astype(str).str.strip().str.lower().map(capped_map).fillna(1)

    # --- Base price in crores (₹1 Cr = 10,000,000) ---
    df['BasePrice_Cr'] = df['Base Price'].fillna(0) / 1e7

    # --- Per-player historical stats (computed only from PAST auctions) ---
    # We use expanding window to avoid data leakage
    player_stats = []

    for idx, row in df.iterrows():
        p = row['Player']
        yr = row['Year']
        # All previous auction appearances for this player (strictly before this year)
        history = df[(df['Player'] == p) & (df['Year'] < yr)]

        if len(history) == 0:
            # First appearance: use base price as proxy
            avg_bid       = row['Base Price'] if not np.isnan(row['Base Price']) else 0
            max_bid       = avg_bid
            min_bid       = avg_bid
            num_auctions  = 0
            last_bid      = avg_bid
            bid_growth    = 0.0
            avg_multiplier= 0.0
        else:
            bids = history['Winning Bid'].values
            bases = history['Base Price'].fillna(1).values
            avg_bid       = float(np.mean(bids))
            max_bid       = float(np.max(bids))
            min_bid       = float(np.min(bids))
            num_auctions  = len(history)
            last_bid      = float(history.sort_values('Year').iloc[-1]['Winning Bid'])

            # Growth: slope of bid over year (simple linear trend)
            years = history['Year'].values.astype(float)
            if len(years) > 1 and np.std(years) > 0:
                bid_growth = float(np.polyfit(years, bids, 1)[0])
            else:
                bid_growth = 0.0

            multipliers = bids / np.maximum(bases, 1)
            avg_multiplier = float(np.mean(multipliers))

        player_stats.append({
            'idx':          idx,
            'hist_avg_bid': avg_bid,
            'hist_max_bid': max_bid,
            'hist_min_bid': min_bid,
            'hist_n_auctions': num_auctions,
            'hist_last_bid': last_bid,
            'hist_bid_growth': bid_growth,
            'hist_avg_multiplier': avg_multiplier,
        })

    stats_df = pd.DataFrame(player_stats).set_index('idx')
    df = df.join(stats_df)

    # --- Scale historical bids to crores ---
    for col in ['hist_avg_bid', 'hist_max_bid', 'hist_min_bid', 'hist_last_bid', 'hist_bid_growth']:
        df[col] = df[col] / 1e7

    # --- Year as a feature (captures IPL's inflation over time) ---
    df['Year_norm'] = (df['Year'] - 2013) / 12.0   # 0 to 1 scale

    # --- Target in crores ---
    df['target'] = df['Winning Bid'] / 1e7

    return df

# ──────────────────────────────────────────────
# 3.  TRAIN STACKING REGRESSOR
# ──────────────────────────────────────────────

FEATURES = [
    'BasePrice_Cr',
    'Overseas_enc',
    'RTM_enc',
    'Capped_enc',
    'Year_norm',
    'hist_avg_bid',
    'hist_max_bid',
    'hist_last_bid',
    'hist_bid_growth',
    'hist_avg_multiplier',
    'hist_n_auctions',
]

def build_stacking_model():
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=8,
                                     min_samples_leaf=3, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, random_state=42)),
    ]
    if HAS_XGB:
        base_estimators.append(
            ('xgb', XGBRegressor(n_estimators=200, max_depth=5,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42,
                                  verbosity=0))
        )
    final = Ridge(alpha=1.0)
    return StackingRegressor(estimators=base_estimators, final_estimator=final,
                             cv=5, n_jobs=-1)

def train():
    print("=" * 55)
    print("  IPL Auction Price Prediction — Model Training")
    print("=" * 55)

    # 1. Load
    print("\n[1] Loading data...")
    df = load_all_data()
    print(f"    {len(df)} valid auction records, {df['Player'].nunique()} unique players")

    # 2. Feature engineering
    print("\n[2] Engineering features (this may take ~30s)...")
    df = engineer_features(df)

    X = df[FEATURES].fillna(0)
    y = df['target']   # Winning Bid in ₹ Crores

    print(f"    Feature matrix: {X.shape}")
    print(f"    Target range: ₹{y.min():.2f}Cr  →  ₹{y.max():.2f}Cr")

    # 3. Build and evaluate with cross-validation
    print("\n[3] Building Stacking Regressor...")
    if HAS_XGB:
        print("    ✓ Using: RandomForest + GradientBoosting + XGBoost → Ridge")
    else:
        print("    ✓ Using: RandomForest + GradientBoosting → Ridge  (XGBoost not installed)")

    model = build_stacking_model()

    print("\n[4] Cross-validating (5-fold)...")
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    cv_mae = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    print(f"    CV R²  : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    print(f"    CV MAE : ₹{(-cv_mae.mean()):.2f}Cr ± {cv_mae.std():.2f}Cr")

    # 4. Fit on full data
    print("\n[5] Fitting final model on all data...")
    model.fit(X, y)

    train_pred = model.predict(X)
    print(f"    Train R²  : {r2_score(y, train_pred):.3f}")
    print(f"    Train MAE : ₹{mean_absolute_error(y, train_pred):.2f}Cr")

    # 5. Quick sanity check
    print("\n[6] Sanity check — sample predictions:")
    test_cases = [
        # (label, BasePrice_Cr, overseas, rtm, capped_enc, year, hist_avg, hist_max, hist_last, growth, mult, n_auctions)
        ("Unknown uncapped rookie (2024)",    0.20, 0, 0, 0, 2024, 0.20, 0.20, 0.20, 0.0, 0.0, 0),
        ("Decent domestic allrounder (2024)", 0.50, 0, 0, 2, 2024, 1.50, 2.00, 1.80, 0.1, 3.0, 2),
        ("Good overseas pacer (2024)",        2.00, 1, 0, 2, 2024, 6.00, 8.00, 7.50, 0.3, 3.5, 3),
        ("Star Indian batsman (2025)",        2.00, 0, 0, 2, 2025, 12.0, 16.0, 14.0, 0.8, 7.0, 4),
        ("Elite overseas all-rounder (2025)", 2.00, 1, 0, 2, 2025, 14.0, 18.0, 16.0, 1.0, 8.0, 5),
    ]
    for label, bp, ov, rtm, cap, yr, havg, hmax, hlast, hgrowth, hmult, nauc in test_cases:
        row = pd.DataFrame([{
            'BasePrice_Cr': bp,
            'Overseas_enc': ov,
            'RTM_enc': rtm,
            'Capped_enc': cap,
            'Year_norm': (yr - 2013) / 12.0,
            'hist_avg_bid': havg,
            'hist_max_bid': hmax,
            'hist_last_bid': hlast,
            'hist_bid_growth': hgrowth,
            'hist_avg_multiplier': hmult,
            'hist_n_auctions': nauc,
        }])
        pred = model.predict(row)[0]
        print(f"    {label:45s}  → ₹{pred:.2f}Cr")

    # 6. Save
    print("\n[7] Saving model artifacts...")
    artifact = {
        'model': model,
        'features': FEATURES,
        'feature_descriptions': {
            'BasePrice_Cr':        'Player base price in ₹ Crores',
            'Overseas_enc':        '1 if overseas player, 0 if domestic',
            'RTM_enc':             '1 if bought via Right to Match, 0 otherwise',
            'Capped_enc':          '2=Capped, 1=Unknown, 0=Uncapped',
            'Year_norm':           'Auction year normalised (0=2013, 1=2025)',
            'hist_avg_bid':        "Player's average historical winning bid (₹Cr)",
            'hist_max_bid':        "Player's peak historical winning bid (₹Cr)",
            'hist_last_bid':       "Player's most recent auction price (₹Cr)",
            'hist_bid_growth':     "Year-on-year trend in bids (₹Cr/year)",
            'hist_avg_multiplier': "Average ratio of winning bid to base price",
            'hist_n_auctions':     "Number of previous IPL auction appearances",
        },
        'target_unit': 'INR Crores',
        'version': '2.0',
    }
    joblib.dump(artifact, 'ipl_auction_model.pkl')
    print("    ✓ Saved → ipl_auction_model.pkl")
    print("\n✅ Training complete!\n")

if __name__ == '__main__':
    train()
