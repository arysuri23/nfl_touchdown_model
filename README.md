# NFL Touchdown Prediction for Betting 🏈

A comprehensive machine learning system for predicting NFL touchdown scorers, optimized for sports betting applications. The project uses advanced feature engineering, multiple model types, and precision-focused optimization to identify profitable betting opportunities.

## 🎯 Project Goals

**Primary Objective**: Build a machine learning system that predicts which NFL players will score touchdowns in upcoming games, specifically optimized for betting profitability.

**Key Requirements**:
- **Precision-optimized models** - Minimize false positives (losing bets)
- **Position-specific predictions** - Separate models for RB, WR/TE, and QB
- **Data leakage prevention** - Proper time-based validation and feature engineering
- **Real-time applicability** - Ready for 2025 season predictions

## 📊 Current Status: COMPLETE MODEL TRAINING PHASE

### ✅ What We've Implemented

#### 1. **Data Collection & Engineering** (`data_collection.py`)
- **Multi-source data integration**: NFL play-by-play, player stats, depth charts, snap counts
- **5 years of historical data**: 2020-2024 (26,236 player-week observations)
- **Comprehensive feature set**: 99 raw features covering usage, efficiency, matchups, and betting lines
- **Team mapping system**: Handles team name variations across data sources

#### 2. **Advanced Feature Engineering** (`feature_engineering.py`)
- **Rolling averages with data leakage prevention**: 3-week rolling windows with `.shift(1)`
- **Position-specific features**: Tailored for RB (22 features), WR/TE (24 features), QB (14 features)
- **Matchup interaction features**: Player usage × opponent defensive weakness
- **Team-level defensive rolling averages**: Consistent opponent stats for all players facing same team
- **Cross-season carryover**: Allows rolling averages to persist between seasons for continuity

**Key Features by Position**:
- **RB**: Snap share, rushing EPA, redzone carries, time to line of scrimmage
- **WR/TE**: Target share, WOPR, air yards, receiving EPA, redzone targets  
- **QB**: Rushing yards/carries, passing EPA, redzone usage (focuses on rushing TDs)

#### 3. **Model Training & Comparison** (`model.py`)
- **Three model types**: Random Forest, XGBoost, LightGBM
- **Precision-optimized hyperparameter tuning**: Uses RandomizedSearchCV with TimeSeriesSplit
- **Time-based data splitting**: 2020-2023 train, 2024 test (prevents data leakage)
- **Comprehensive evaluation**: AUC-ROC, precision, recall, confusion matrices
- **Model persistence**: Saved using joblib for production deployment

#### 4. **Data Quality & Validation**
- **Deduplication**: Ensures one record per player-week
- **Missing value handling**: Strategic filling with zeros
- **Cross-season isolation**: Prevents data leakage between seasons when desired
- **Opponent feature consistency**: All players facing same opponent get identical defensive stats

### 🏆 Model Performance Results

**Final Model Rankings (AUC-ROC)**:

| Position | 🥇 Random Forest | 🥈 Second Place | 🥉 Third Place |
|----------|------------------|-----------------|----------------|
| **RB** | **0.694** | LightGBM (0.686) | XGBoost (0.674) |
| **WR/TE** | **0.656** | XGBoost (0.635) | LightGBM (0.602) |
| **QB** | **0.697** | XGBoost (0.660) | LightGBM (0.651) |

**Betting Performance (Precision)**:
- **RB**: 47% precision (when we predict TD, we're right 47% of the time)
- **WR/TE**: 36% precision (more conservative, fewer false positives)
- **QB**: 28% precision (lowest precision but highest overall AUC)

**Winner**: **Random Forest dominates all positions** - recommended for betting strategy.

## 🗂️ Project Structure

```
td-scorers/
├── README.md                          # This file
├── data_collection.py                 # Multi-source NFL data collection
├── feature_engineering.py             # Advanced feature creation & processing
├── model.py                          # Model training, comparison & evaluation
├── training_data_2020_2024.csv       # Complete engineered dataset
├── 2024_data.csv                     # 2024 season validation data
└── data/                             # Raw data files
    ├── historic_lines.csv            # Historical betting lines
    ├── stats_player_week_2025.csv     # Current season data
    ├── week_1_lines.csv              # Weekly betting lines
    ├── week_1_td_odds.csv            # TD scorer odds
    └── ...
```

## 🔧 Key Functions & Usage

### Data Pipeline
```python
# Generate complete training dataset
from model import generate_training_data
datasets, combined_data = generate_training_data()

# Create 2024 validation data
from feature_engineering import prepare_touchdown_prediction_data
data_2024 = prepare_touchdown_prediction_data([2024])
```

### Model Training & Comparison
```python
# Train and compare all models
from model import train_and_compare_all_models
results = train_and_compare_all_models(tune_hyperparams=True)

# Train individual models
from model import train_random_forest_model, train_xgboost_model, train_lightgbm_model
rb_model = train_random_forest_model(position='RB')
```

### Feature Engineering
```python
# Create engineered features
from feature_engineering import engineer_features_for_touchdown_prediction
df = engineer_features_for_touchdown_prediction(years=[2020,2021,2022,2023,2024])

# Create position-specific datasets
from feature_engineering import create_position_specific_datasets
position_datasets = create_position_specific_datasets(df)
```

## 📈 Dataset Statistics

- **Total Observations**: 26,236 player-week records
- **Time Period**: 2020-2024 NFL seasons (5 years)
- **Total Touchdowns**: 5,494 (20.9% overall rate)
- **Position Breakdown**:
  - RB: 6,807 samples, 26.9% TD rate
  - WR/TE: 16,153 samples, 19.7% TD rate  
  - QB: 3,276 samples, 14.5% TD rate

**Data Validation**:
- ✅ No data leakage between train/test splits
- ✅ Rolling averages properly shifted to prevent future information
- ✅ Consistent opponent features for all players facing same team
- ✅ Cross-season carryover allowed for realistic modeling

## 🧠 Technical Implementation Details

### Feature Engineering Pipeline
1. **Raw data loading** from multiple NFL data sources
2. **Player-level rolling averages** (3-week windows with lag)
3. **Team-level defensive rolling averages** (opponent-specific)
4. **Matchup interaction features** (usage × defensive weakness)
5. **Position-specific feature selection** and filtering
6. **Data quality validation** and deduplication

### Model Training Process
1. **Time-based data splitting** (2020-2023 train, 2024 test)
2. **Hyperparameter optimization** using RandomizedSearchCV
3. **Cross-validation** with TimeSeriesSplit (maintains temporal order)
4. **Precision optimization** (critical for betting profitability)
5. **Model persistence** using joblib for deployment

### Key Technical Decisions
- **Precision over recall**: Optimized for betting (avoid false positives)
- **Position-specific models**: Different features matter for different positions
- **Rolling averages**: 3-week windows balance recency vs. stability
- **Cross-season carryover**: Maintains player/team continuity
- **Random Forest winner**: Best balance of performance and interpretability

## 🚀 Next Steps & Future Development

### Phase 1: Production Deployment (Immediate)
- [ ] **2025 Prediction Pipeline**: Build system to generate weekly predictions
- [ ] **Model Loading System**: Efficient loading of trained Random Forest models
- [ ] **Confidence Thresholding**: Only bet on high-confidence predictions
- [ ] **Weekly Data Updates**: Automated data collection for new weeks

### Phase 2: Strategy Enhancement
- [ ] **Ensemble Methods**: Combine Random Forest + LightGBM for RB positions
- [ ] **Betting Line Integration**: Compare model predictions to actual TD odds
- [ ] **Bankroll Management**: Implement Kelly Criterion or similar strategies
- [ ] **Performance Tracking**: Monitor actual betting results vs. predictions

### Phase 3: Advanced Features
- [ ] **Weather Integration**: Add weather data for outdoor games
- [ ] **Injury Reports**: Incorporate player injury status
- [ ] **Game Script Prediction**: Predict game flow to improve TD probabilities
- [ ] **Real-time Updates**: Intra-game model updates based on game state

### Phase 4: Model Improvements
- [ ] **Neural Networks**: Experiment with deep learning approaches
- [ ] **Feature Selection**: Advanced feature importance and selection methods
- [ ] **Temporal Modeling**: LSTM/RNN for sequence-based predictions
- [ ] **Multi-target Prediction**: Predict multiple TD scorers per game

## 💡 Key Insights & Lessons Learned

### Data Quality is Critical
- **Deduplication essential**: Multiple data sources create duplicate records
- **Opponent feature consistency**: Manual validation revealed inconsistencies
- **Rolling average validation**: Extensive testing prevented data leakage

### Model Selection Findings
- **Random Forest superiority**: Consistently outperformed gradient boosting methods
- **Position-specific optimization**: Different models work better for different positions
- **Precision focus**: Betting applications require precision over recall optimization

### Feature Engineering Impact
- **Rolling averages crucial**: Recent performance is highly predictive
- **Matchup features powerful**: Interaction between usage and opponent weakness
- **Position-specific features**: Tailored features significantly improve performance

## 🎯 Business Value

**For Sports Betting**:
- **Precision-optimized models** minimize losing bets
- **Position-specific predictions** allow targeted betting strategies
- **Historical validation** on 2024 season provides confidence
- **Ready for 2025 deployment** with established infrastructure

**Expected Performance**:
- **RB bets**: ~47% success rate on predicted touchdowns
- **WR/TE bets**: ~36% success rate (more conservative)
- **QB bets**: ~28% success rate (focus on rushing TDs)

## 📚 Dependencies

```python
# Core ML & Data
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.6.0
lightgbm>=3.3.0

# Data Collection
nfl_data_py>=0.3.0
requests>=2.28.0

# Model Persistence
joblib>=1.1.0

# Utilities
datetime
os
```

## 🔄 Running the Complete Pipeline

```bash
# Train all models and compare performance
python model.py

# Generate 2024 validation data
python -c "from feature_engineering import prepare_touchdown_prediction_data; prepare_touchdown_prediction_data([2024])"

# Quick model test
python -c "from model import train_random_forest_model; train_random_forest_model(position='RB')"
```

---

**Project Status**: ✅ **MODEL TRAINING COMPLETE** - Ready for production deployment and 2025 season predictions.

**Contact**: Built for sports betting applications with a focus on precision and profitability.
