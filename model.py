import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from feature_engineering import prepare_touchdown_prediction_data


def generate_training_data():
    """
    Generate comprehensive training dataset from 2020-2024 for touchdown prediction models.
    
    Returns:
        dict: Position-specific datasets (RB, WR_TE, QB) with combined data from 2020-2024
    """
    
    # Team mapping for consistent data processing
    team_map = {
        'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LVR', 'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
        'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
    }
    
    print("🏈 GENERATING COMPREHENSIVE TRAINING DATASET (2020-2024)")
    print("=" * 65)
    print()
    
    # Generate multi-year dataset using our feature engineering pipeline
    years = [2020, 2021, 2022, 2023, 2024]
    print(f"Loading data for years: {years}")
    
    datasets = prepare_touchdown_prediction_data(years, team_map, window=5)
    
    # Combine all position datasets with position group labels
    print("\n📊 Combining position datasets...")
    all_data = pd.concat([
        datasets['RB'].assign(position_group='RB'),
        datasets['WR_TE'].assign(position_group='WR_TE'),
        datasets['QB'].assign(position_group='QB')
    ], ignore_index=True)
    
    # Save comprehensive training dataset
    output_file = 'training_data_2020_2024.csv'
    all_data.to_csv(output_file, index=False)
    
    print(f"\n✅ TRAINING DATASET GENERATED")
    print(f"📁 Saved to: {output_file}")
    print(f"📈 Total samples: {all_data.shape[0]:,}")
    print(f"📈 Total features: {all_data.shape[1]}")
    print(f"📈 Total touchdowns: {all_data['scored_touchdown'].sum():,}")
    print(f"📈 TD rate: {all_data['scored_touchdown'].mean():.1%}")
    
    # Show breakdown by position and year
    print(f"\n📊 BREAKDOWN BY POSITION:")
    for pos in ['RB', 'WR_TE', 'QB']:
        pos_data = all_data[all_data['position_group'] == pos]
        td_rate = pos_data['scored_touchdown'].mean()
        print(f"  • {pos}: {len(pos_data):,} samples, {td_rate:.1%} TD rate")
    
    print(f"\n📊 BREAKDOWN BY YEAR:")
    for year in years:
        year_data = all_data[all_data['season'] == year]
        td_rate = year_data['scored_touchdown'].mean()
        print(f"  • {year}: {len(year_data):,} samples, {td_rate:.1%} TD rate")
    
    print(f"\n🎯 Ready for time-based splitting and model training!")
    
    return datasets, all_data


def create_time_based_splits(data_file='training_data_2020_2024.csv'):
    """
    Create time-based train/test splits for touchdown prediction.
    
    Split strategy:
    - Train: 2020-2023 (4 full seasons) - use cross-validation for hyperparameter tuning
    - Test: 2024 (full season) - final evaluation on most recent data
    
    Args:
        data_file: Path to the training dataset CSV
        
    Returns:
        dict: Contains train and test datasets
    """
    
    print("📊 CREATING TIME-BASED DATA SPLITS")
    print("=" * 45)
    print()
    
    # Load the comprehensive training dataset
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    
    print(f"Total samples loaded: {len(df):,}")
    print(f"Date range: {df['season'].min()}-{df['season'].max()}")
    print(f"Week range: {df['week'].min()}-{df['week'].max()}")
    
    # Create simple time-based splits
    train_data = df[df['season'] <= 2023].copy()
    test_data = df[df['season'] == 2024].copy()
    
    # Summary statistics
    splits = {
        'train': train_data,
        'test': test_data
    }
    
    print(f"\n📈 SPLIT SUMMARY:")
    for split_name, split_data in splits.items():
        td_count = split_data['scored_touchdown'].sum()
        td_rate = split_data['scored_touchdown'].mean()
        seasons = sorted(split_data['season'].unique())
        weeks = f"{split_data['week'].min()}-{split_data['week'].max()}"
        
        print(f"  • {split_name.upper()}: {len(split_data):,} samples")
        print(f"    - Touchdowns: {td_count:,} ({td_rate:.1%} rate)")
        print(f"    - Seasons: {seasons}")
        print(f"    - Weeks: {weeks}")
        print()
    
    # Position breakdown for each split
    print(f"📊 POSITION BREAKDOWN BY SPLIT:")
    for split_name, split_data in splits.items():
        print(f"\n  {split_name.upper()}:")
        for pos in ['RB', 'WR_TE', 'QB']:
            pos_data = split_data[split_data['position_group'] == pos]
            if len(pos_data) > 0:
                td_rate = pos_data['scored_touchdown'].mean()
                print(f"    - {pos}: {len(pos_data):,} samples, {td_rate:.1%} TD rate")
    
    print(f"\n✅ Time-based splits created successfully!")
    print(f"✅ Clean year-based split: 4 seasons train, 1 season test")
    print(f"✅ Use cross-validation on train data for hyperparameter tuning")
    print(f"✅ Ready for model training!")
    
    return splits


def train_random_forest_model(position='RB', data_file='training_data_2020_2024.csv', tune_hyperparams=True):
    """
    Train a Random Forest model for touchdown prediction with optional hyperparameter tuning.
    
    Args:
        position: Position to train model for ('RB', 'WR_TE', 'QB')
        data_file: Path to the training dataset CSV
        tune_hyperparams: Whether to perform hyperparameter tuning (default True)
        
    Returns:
        dict: Contains trained model, feature names, and evaluation metrics
    """
    
    print(f"🌲 TRAINING RANDOM FOREST MODEL - {position}")
    print("=" * 50)
    print()
    
    # Create train/test splits
    print("Loading and splitting data...")
    splits = create_time_based_splits(data_file)
    
    # Filter for specific position
    train_data = splits['train'][splits['train']['position_group'] == position].copy()
    test_data = splits['test'][splits['test']['position_group'] == position].copy()
    
    print(f"Position: {position}")
    print(f"Train samples: {len(train_data):,}")
    print(f"Test samples: {len(test_data):,}")
    print(f"Train TD rate: {train_data['scored_touchdown'].mean():.1%}")
    print(f"Test TD rate: {test_data['scored_touchdown'].mean():.1%}")
    print()
    
    # Prepare features and target
    # Remove non-feature columns
    feature_columns = [col for col in train_data.columns if col not in [
        'player_id', 'player_display_name', 'position', 'recent_team', 
        'opponent_team', 'season', 'week', 'scored_touchdown', 'position_group'
    ]]
    
    X_train = train_data[feature_columns]
    y_train = train_data['scored_touchdown']
    X_test = test_data[feature_columns]
    y_test = test_data['scored_touchdown']
    
    print(f"Features used: {len(feature_columns)}")
    print(f"Feature names: {feature_columns[:5]}... (showing first 5)")
    print()
    
    # Handle missing values (fill with 0)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Train Random Forest model with optional hyperparameter tuning
    if tune_hyperparams:
        print("🔍 Performing hyperparameter tuning with Randomized Search + Time Series CV...")
        
        # Define parameter distributions for random sampling
        param_distributions = {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [8, 10, 12, 15, None],
            'min_samples_split': [10, 15, 20, 30, 50],
            'min_samples_leaf': [5, 10, 15, 20, 25],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Base model
        rf_base = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation strategy (time series to maintain temporal order)
        cv_strategy = TimeSeriesSplit(n_splits=3)
        
        # Randomized search optimized for PR-AUC (ranking in imbalanced data)
        random_search = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=param_distributions,
            n_iter=50,                    # Try 50 random combinations (vs 324 in GridSearch)
            cv=cv_strategy,
            scoring='average_precision',  # Optimize PR-AUC
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # Fit randomized search
        random_search.fit(X_train, y_train)
        
        # Best params (we will refit for calibration below)
        best_params = random_search.best_estimator_.get_params()
        
        print(f"✅ Hyperparameter tuning completed!")
        print(f"Best CV PR-AUC (Average Precision): {random_search.best_score_:.3f}")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Tested {random_search.n_iter} parameter combinations (vs 324 in full grid)")
        print(f"🎯 Optimized for BETTING: Higher PR-AUC for better ranked picks!")
        print()
        
    else:
        print("Training Random Forest with default parameters...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        best_params = rf_model.get_params()
        print("✅ Default parameters prepared (no tuning)")
        print()
    # ==== Probability calibration using time-based split ====
    print("Calibrating probabilities with time-based split (isotonic)...")
    train_sorted = train_data.sort_values(["season", "week"]).reset_index(drop=True)
    fit_mask = train_sorted["season"] <= 2022
    cal_mask = train_sorted["season"] == 2023
    if cal_mask.sum() == 0 or fit_mask.sum() == 0:
        split_idx = int(len(train_sorted) * 0.8)
        fit_df = train_sorted.iloc[:split_idx]
        cal_df = train_sorted.iloc[split_idx:]
    else:
        fit_df = train_sorted[fit_mask]
        cal_df = train_sorted[cal_mask]

    X_fit = fit_df[feature_columns].fillna(0)
    y_fit = fit_df['scored_touchdown']
    X_cal = cal_df[feature_columns].fillna(0)
    y_cal = cal_df['scored_touchdown']

    rf_prefit = RandomForestClassifier(**best_params)
    rf_prefit.fit(X_fit, y_fit)
    calibrator = CalibratedClassifierCV(rf_prefit, method='isotonic', cv='prefit')
    calibrator.fit(X_cal, y_cal)

    print("Generating predictions (calibrated)...")
    y_pred = calibrator.predict(X_test)
    y_pred_proba = calibrator.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print("📊 MODEL EVALUATION RESULTS:")
    print("-" * 35)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No TD', 'TD']))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives: {cm[0,0]:,}, False Positives: {cm[0,1]:,}")
    print(f"False Negatives: {cm[1,0]:,}, True Positives: {cm[1,1]:,}")
    print()
    
    # AUC-ROC and PR-AUC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score: {auc_score:.3f}")
    print(f"PR-AUC (Average Precision): {ap_score:.3f}")
    print()
    
    # Feature importance (top 10)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_prefit.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🔥 TOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 40)
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    print(f"\n✅ {position} Random Forest model training completed!")
    print(f"✅ Model ready for touchdown predictions!")
    
    model_results = {
        'model': calibrator,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance,
        'test_auc': auc_score,
        'test_predictions': y_pred_proba,
        'test_ap': ap_score,
        'is_calibrated': True,
        'calibration_method': 'isotonic'
    }
    
    # Save the trained model
    model_path = save_model(model_results, position, model_type='RandomForest')
    model_results['model_path'] = model_path
    
    return model_results


def train_xgboost_model(position='RB', data_file='training_data_2020_2024.csv', tune_hyperparams=True):
    """
    Train XGBoost model for touchdown prediction, optimized for betting precision.
    
    Args:
        position: Position to train model for ('RB', 'WR_TE', 'QB')
        data_file: Path to training data CSV file
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        dict: Trained model results including model, features, and performance metrics
    """
    
    print(f"🚀 TRAINING XGBOOST MODEL - {position}")
    print("=" * 50)
    print()
    
    # Create train/test splits
    print("Loading and splitting data...")
    splits = create_time_based_splits(data_file)
    
    # Filter for specific position
    train_data = splits['train'][splits['train']['position_group'] == position].copy()
    test_data = splits['test'][splits['test']['position_group'] == position].copy()
    
    print(f"Position: {position}")
    print(f"Train samples: {len(train_data):,}")
    print(f"Test samples: {len(test_data):,}")
    print(f"Train TD rate: {train_data['scored_touchdown'].mean():.1%}")
    print(f"Test TD rate: {test_data['scored_touchdown'].mean():.1%}")
    print()
    
    # Prepare features and target
    # Remove non-feature columns
    feature_columns = [col for col in train_data.columns if col not in [
        'player_id', 'player_display_name', 'position', 'recent_team', 
        'opponent_team', 'season', 'week', 'scored_touchdown', 'position_group'
    ]]
    
    X_train = train_data[feature_columns]
    y_train = train_data['scored_touchdown']
    X_test = test_data[feature_columns]
    y_test = test_data['scored_touchdown']
    
    print(f"Features used: {len(feature_columns)}")
    print(f"Feature names: {feature_columns[:5]}... (showing first 5)")
    print()
    
    # Handle missing values (fill with 0)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    if tune_hyperparams:
        print("🔍 Performing hyperparameter tuning with Randomized Search + Time Series CV...")
        
        # XGBoost hyperparameter search space
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0]
        }
        
        # Create XGBoost classifier with class balancing
        xgb_base = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle class imbalance
        )
        
        # Time series cross-validation for hyperparameter tuning
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Randomized search optimized for PR-AUC (betting/ranking focus)
        random_search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_distributions,
            n_iter=50,  # Test 50 random combinations
            cv=tscv,
            scoring='average_precision',  # Optimize PR-AUC
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the randomized search
        random_search.fit(X_train, y_train)
        
        # Best params (will refit for calibration below)
        best_params = random_search.best_estimator_.get_params()
        
        print("✅ Hyperparameter tuning completed!")
        print(f"Best CV PR-AUC (Average Precision): {random_search.best_score_:.3f}")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Tested {len(random_search.cv_results_['params'])} parameter combinations")
        print("🎯 Optimized for BETTING: Higher PR-AUC for better ranked picks!")
        
    else:
        # Use default XGBoost with class balancing
        print("Using default XGBoost parameters with class balancing...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
        )
        best_params = xgb_model.get_params()
    
    print()
    # ==== Probability calibration using time-based split ====
    print("Calibrating probabilities with time-based split (isotonic)...")
    train_sorted = train_data.sort_values(["season", "week"]).reset_index(drop=True)
    fit_mask = train_sorted["season"] <= 2022
    cal_mask = train_sorted["season"] == 2023
    if cal_mask.sum() == 0 or fit_mask.sum() == 0:
        split_idx = int(len(train_sorted) * 0.8)
        fit_df = train_sorted.iloc[:split_idx]
        cal_df = train_sorted.iloc[split_idx:]
    else:
        fit_df = train_sorted[fit_mask]
        cal_df = train_sorted[cal_mask]

    X_fit = fit_df[feature_columns].fillna(0)
    y_fit = fit_df['scored_touchdown']
    X_cal = cal_df[feature_columns].fillna(0)
    y_cal = cal_df['scored_touchdown']

    xgb_prefit = xgb.XGBClassifier(**best_params)
    xgb_prefit.fit(X_fit, y_fit)
    calibrator = CalibratedClassifierCV(xgb_prefit, method='isotonic', cv='prefit')
    calibrator.fit(X_cal, y_cal)

    print("Generating predictions (calibrated)...")
    y_pred = calibrator.predict(X_test)
    y_pred_proba = calibrator.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    
    # Print evaluation results
    print("📊 MODEL EVALUATION RESULTS:")
    print("-----------------------------------")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No TD', 'TD']))
    
    print("Confusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True Negatives: {tn:,}, False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}, True Positives: {tp:,}")
    
    print(f"\nAUC-ROC Score: {auc_score:.3f}")
    print(f"PR-AUC (Average Precision): {ap_score:.3f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': xgb_prefit.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print()
    print("🔥 TOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 40)
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    print(f"\n✅ {position} XGBoost model training completed!")
    print(f"✅ Model ready for touchdown predictions!")
    
    model_results = {
        'model': calibrator,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance,
        'test_auc': auc_score,
        'test_ap': ap_score,
        'test_predictions': y_pred_proba,
        'is_calibrated': True,
        'calibration_method': 'isotonic'
    }
    
    # Save the trained model
    model_path = save_model(model_results, position, model_type='XGBoost')
    model_results['model_path'] = model_path
    
    return model_results


def train_lightgbm_model(position='RB', data_file='training_data_2020_2024.csv', tune_hyperparams=True):
    """
    Train LightGBM model for touchdown prediction, optimized for betting precision.
    
    Args:
        position: Position to train model for ('RB', 'WR_TE', 'QB')
        data_file: Path to training data CSV file
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        dict: Trained model results including model, features, and performance metrics
    """
    
    print(f"💡 TRAINING LIGHTGBM MODEL - {position}")
    print("=" * 50)
    print()
    
    # Create train/test splits
    print("Loading and splitting data...")
    splits = create_time_based_splits(data_file)
    
    # Filter for specific position
    train_data = splits['train'][splits['train']['position_group'] == position].copy()
    test_data = splits['test'][splits['test']['position_group'] == position].copy()
    
    print(f"Position: {position}")
    print(f"Train samples: {len(train_data):,}")
    print(f"Test samples: {len(test_data):,}")
    print(f"Train TD rate: {train_data['scored_touchdown'].mean():.1%}")
    print(f"Test TD rate: {test_data['scored_touchdown'].mean():.1%}")
    print()
    
    # Prepare features and target
    # Remove non-feature columns
    feature_columns = [col for col in train_data.columns if col not in [
        'player_id', 'player_display_name', 'position', 'recent_team', 
        'opponent_team', 'season', 'week', 'scored_touchdown', 'position_group'
    ]]
    
    X_train = train_data[feature_columns]
    y_train = train_data['scored_touchdown']
    X_test = test_data[feature_columns]
    y_test = test_data['scored_touchdown']
    
    print(f"Features used: {len(feature_columns)}")
    print(f"Feature names: {feature_columns[:5]}... (showing first 5)")
    print()
    
    # Handle missing values (fill with 0)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    if tune_hyperparams:
        print("🔍 Performing hyperparameter tuning with Randomized Search + Time Series CV...")
        
        # LightGBM hyperparameter search space
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],
            'num_leaves': [31, 50, 100, 150],
            'min_child_samples': [10, 20, 30]
        }
        
        # Create LightGBM classifier with class balancing
        lgb_base = lgb.LGBMClassifier(
            random_state=42,
            objective='binary',
            metric='binary_logloss',
            verbosity=-1,  # Suppress warnings
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Time series cross-validation for hyperparameter tuning
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Randomized search optimized for PR-AUC (betting/ranking focus)
        random_search = RandomizedSearchCV(
            estimator=lgb_base,
            param_distributions=param_distributions,
            n_iter=50,  # Test 50 random combinations
            cv=tscv,
            scoring='average_precision',  # Optimize PR-AUC
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the randomized search
        random_search.fit(X_train, y_train)
        
        # Best params (will refit for calibration below)
        best_params = random_search.best_estimator_.get_params()
        
        print("✅ Hyperparameter tuning completed!")
        print(f"Best CV PR-AUC (Average Precision): {random_search.best_score_:.3f}")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Tested {len(random_search.cv_results_['params'])} parameter combinations")
        print("🎯 Optimized for BETTING: Higher PR-AUC for better ranked picks!")
        
    else:
        # Use default LightGBM with class balancing
        print("Using default LightGBM parameters with class balancing...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective='binary',
            metric='binary_logloss',
            verbosity=-1,
            class_weight='balanced'
        )
        best_params = lgb_model.get_params()
    
    print()
    # ==== Probability calibration using time-based split ====
    print("Calibrating probabilities with time-based split (isotonic)...")
    train_sorted = train_data.sort_values(["season", "week"]).reset_index(drop=True)
    fit_mask = train_sorted["season"] <= 2022
    cal_mask = train_sorted["season"] == 2023
    if cal_mask.sum() == 0 or fit_mask.sum() == 0:
        split_idx = int(len(train_sorted) * 0.8)
        fit_df = train_sorted.iloc[:split_idx]
        cal_df = train_sorted.iloc[split_idx:]
    else:
        fit_df = train_sorted[fit_mask]
        cal_df = train_sorted[cal_mask]

    X_fit = fit_df[feature_columns].fillna(0)
    y_fit = fit_df['scored_touchdown']
    X_cal = cal_df[feature_columns].fillna(0)
    y_cal = cal_df['scored_touchdown']

    lgb_prefit = lgb.LGBMClassifier(**best_params)
    lgb_prefit.fit(X_fit, y_fit)
    calibrator = CalibratedClassifierCV(lgb_prefit, method='isotonic', cv='prefit')
    calibrator.fit(X_cal, y_cal)

    print("Generating predictions (calibrated)...")
    y_pred = calibrator.predict(X_test)
    y_pred_proba = calibrator.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    
    # Print evaluation results
    print("📊 MODEL EVALUATION RESULTS:")
    print("-----------------------------------")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No TD', 'TD']))
    
    print("Confusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True Negatives: {tn:,}, False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}, True Positives: {tp:,}")
    
    print(f"\nAUC-ROC Score: {auc_score:.3f}")
    print(f"PR-AUC (Average Precision): {ap_score:.3f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': lgb_prefit.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print()
    print("🔥 TOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 40)
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    print(f"\n✅ {position} LightGBM model training completed!")
    print(f"✅ Model ready for touchdown predictions!")
    
    model_results = {
        'model': calibrator,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance,
        'test_auc': auc_score,
        'test_ap': ap_score,
        'test_predictions': y_pred_proba,
        'is_calibrated': True,
        'calibration_method': 'isotonic'
    }
    
    # Save the trained model
    model_path = save_model(model_results, position, model_type='LightGBM')
    model_results['model_path'] = model_path
    
    return model_results


def save_model(model_results, position, model_type='RandomForest', model_dir='models'):
    """
    Save trained model and associated metadata to disk.
    
    Args:
        model_results: Dictionary containing model and metadata from training function
        position: Position name ('RB', 'WR_TE', 'QB')
        model_type: Type of model ('RandomForest', 'XGBoost', etc.)
        model_dir: Directory to save models (default 'models')
    """
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Define file paths with clean naming format
    model_filename = f'{position}_{model_type}_touchdown_model.joblib'
    model_path = os.path.join(model_dir, model_filename)
    
    # Create model package with all necessary components
    model_package = {
        'model': model_results['model'],
        'feature_columns': model_results['feature_columns'],
        'feature_importance': model_results['feature_importance'],
        'test_auc': model_results['test_auc'],
        'test_ap': model_results.get('test_ap'),
        'position': position,
        'training_date': datetime.now(),
        'model_type': f'{model_type}_Betting_Optimized',
        'optimization_metric': 'average_precision',
        'training_years': '2020-2023',
        'test_year': '2024'
    }
    
    # Save model package using joblib (more efficient for sklearn models)
    joblib.dump(model_package, model_path)
    
    print(f"💾 {position} {model_type} model saved to: {model_path}")
    return model_path


def load_model(model_path):
    """
    Load a saved touchdown prediction model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        dict: Model package containing model and metadata
    """
    
    model_package = joblib.load(model_path)
    
    print(f"📂 Model loaded from: {model_path}")
    print(f"   Position: {model_package['position']}")
    print(f"   Training Date: {model_package['training_date']}")
    print(f"   Test AUC: {model_package['test_auc']:.3f}")
    
    return model_package


def train_all_position_models(data_file='training_data_2020_2024.csv', tune_hyperparams=True):
    """
    Train betting-optimized Random Forest models for all positions (RB, WR/TE, QB).
    
    Args:
        data_file: Path to the training dataset CSV
        tune_hyperparams: Whether to perform hyperparameter tuning (default True)
        
    Returns:
        dict: Contains trained models and results for all positions
    """
    
    print('🎯 TRAINING BETTING-OPTIMIZED TOUCHDOWN PREDICTION MODELS')
    print('=' * 65)
    print('Optimizing for PRECISION: When we predict TD, maximize accuracy!')
    print()
    
    results = {}
    positions = ['RB', 'WR_TE', 'QB']
    
    for i, position in enumerate(positions, 1):
        print(f'🏃‍♂️ TRAINING {position} MODEL ({i}/3)...')
        print('-' * 40)
        
        # Train position-specific model
        model_results = train_random_forest_model(
            position=position, 
            data_file=data_file, 
            tune_hyperparams=tune_hyperparams
        )
        
        # Model is already saved in training function
        # Store results
        results[position] = model_results
        
        print(f'✅ {position} model completed and saved!')
        
        if i < len(positions):  # Don't print separator after last model
            print('\n' + '='*65)
            print()
    
    # Final summary
    print('\n' + '='*65)
    print('🎯 ALL MODELS TRAINED - READY FOR BETTING!')
    print('='*65)
    print()
    
    # Summary table
    print('📊 BETTING MODEL PERFORMANCE SUMMARY:')
    print('-' * 60)
    print(f'{"Position":<8} | {"Test AUC":<10} | {"Model File"}')
    print('-' * 60)
    for position, model_result in results.items():
        model_filename = os.path.basename(model_result['model_path'])
        print(f'{position:<8} | {model_result["test_auc"]:<10.3f} | {model_filename}')
    
    print(f'\n💾 All models saved to ./models/ directory')
    print(f'🎯 Models optimized for betting precision!')
    print(f'🎯 Use high-confidence predictions for profitable bets!')
    
    return results


def train_all_xgboost_models(data_file='training_data_2020_2024.csv', tune_hyperparams=True):
    """
    Train betting-optimized XGBoost models for all positions (RB, WR/TE, QB).
    
    Args:
        data_file: Path to training data CSV file
        tune_hyperparams: Whether to perform hyperparameter tuning for each model
        
    Returns:
        dict: Results for each position containing trained models and performance metrics
    """
    
    print('🚀 TRAINING BETTING-OPTIMIZED XGBOOST MODELS')
    print('=' * 65)
    print('Optimizing for PRECISION: When we predict TD, maximize accuracy!')
    print()
    
    positions = ['RB', 'WR_TE', 'QB']
    results = {}
    
    for i, position in enumerate(positions, 1):
        print(f'🚀 TRAINING {position} XGBOOST MODEL ({i}/3)...')
        print('-' * 40)
        
        # Train position-specific model
        model_results = train_xgboost_model(
            position=position, 
            data_file=data_file, 
            tune_hyperparams=tune_hyperparams
        )
        
        # Model is already saved in training function
        # Store results
        results[position] = model_results
        
        print(f'✅ {position} XGBoost model completed and saved!')
        
        if i < len(positions):  # Don't print separator after last model
            print('\n' + '='*65)
            print()
    
    # Final summary
    print('\n' + '='*65)
    print('🚀 ALL XGBOOST MODELS TRAINED - READY FOR BETTING!')
    print('='*65)
    print()
    
    # Summary table
    print('📊 XGBOOST MODEL PERFORMANCE SUMMARY:')
    print('-' * 60)
    print(f'{"Position":<8} | {"Test AUC":<10} | {"Model File"}')
    print('-' * 60)
    for position, model_result in results.items():
        model_filename = os.path.basename(model_result['model_path'])
        print(f'{position:<8} | {model_result["test_auc"]:<10.3f} | {model_filename}')
    
    print(f'\n💾 All XGBoost models saved to ./models/ directory')
    print(f'🚀 XGBoost models optimized for betting precision!')
    print(f'🎯 Use high-confidence predictions for profitable bets!')
    
    return results


def compare_model_performance(rf_results, xgb_results):
    """
    Compare Random Forest vs XGBoost model performance.
    
    Args:
        rf_results: Results dictionary from train_all_position_models()
        xgb_results: Results dictionary from train_all_xgboost_models()
    """
    
    print('⚡ MODEL COMPARISON: RANDOM FOREST vs XGBOOST')
    print('=' * 70)
    print(f'{"Position":<8} | {"RF AUC":<8} | {"XGB AUC":<8} | {"Winner":<10} | {"Improvement"}')
    print('-' * 70)
    
    for position in ['RB', 'WR_TE', 'QB']:
        rf_auc = rf_results[position]['test_auc']
        xgb_auc = xgb_results[position]['test_auc']
        
        if xgb_auc > rf_auc:
            winner = 'XGBoost'
            improvement = f'+{((xgb_auc - rf_auc) / rf_auc * 100):.1f}%'
        elif rf_auc > xgb_auc:
            winner = 'RandomForest'
            improvement = f'-{((rf_auc - xgb_auc) / xgb_auc * 100):.1f}%'
        else:
            winner = 'Tie'
            improvement = '0.0%'
        
        print(f'{position:<8} | {rf_auc:<8.3f} | {xgb_auc:<8.3f} | {winner:<10} | {improvement}')
    
    print()
    print('🎯 Use the better-performing model for each position in your betting strategy!')


def train_and_compare_all_models(data_file='training_data_2020_2024.csv', tune_hyperparams=True):
    """
    Train Random Forest, XGBoost, and LightGBM models for all positions and compare performance.
    
    Args:
        data_file: Path to training data CSV file
        tune_hyperparams: Whether to perform hyperparameter tuning for each model
        
    Returns:
        dict: Results for all model types across all positions
    """
    
    print('🤖 TRAINING & COMPARING ALL MODELS: RF vs XGBOOST vs LIGHTGBM')
    print('=' * 70)
    print('Training all three model types for all positions and comparing performance')
    print()
    
    positions = ['RB', 'WR_TE', 'QB']
    results = {'RandomForest': {}, 'XGBoost': {}, 'LightGBM': {}}
    
    # Train Random Forest models
    print('🌲 TRAINING RANDOM FOREST MODELS')
    print('=' * 50)
    for i, position in enumerate(positions, 1):
        print(f'🌲 Training {position} Random Forest ({i}/3)...')
        rf_results = train_random_forest_model(
            position=position, 
            data_file=data_file, 
            tune_hyperparams=tune_hyperparams
        )
        results['RandomForest'][position] = rf_results
        print(f'✅ {position} Random Forest completed! AUC: {rf_results["test_auc"]:.3f}')
        
        if i < len(positions):
            print()
    
    print('\n' + '='*70)
    print()
    
    # Train XGBoost models
    print('🚀 TRAINING XGBOOST MODELS')
    print('=' * 50)
    for i, position in enumerate(positions, 1):
        print(f'🚀 Training {position} XGBoost ({i}/3)...')
        xgb_results = train_xgboost_model(
            position=position, 
            data_file=data_file, 
            tune_hyperparams=tune_hyperparams
        )
        results['XGBoost'][position] = xgb_results
        print(f'✅ {position} XGBoost completed! AUC: {xgb_results["test_auc"]:.3f}')
        
        if i < len(positions):
            print()
    
    print('\n' + '='*70)
    print()
    
    # Train LightGBM models
    print('💡 TRAINING LIGHTGBM MODELS')
    print('=' * 50)
    for i, position in enumerate(positions, 1):
        print(f'💡 Training {position} LightGBM ({i}/3)...')
        lgb_results = train_lightgbm_model(
            position=position, 
            data_file=data_file, 
            tune_hyperparams=tune_hyperparams
        )
        results['LightGBM'][position] = lgb_results
        print(f'✅ {position} LightGBM completed! AUC: {lgb_results["test_auc"]:.3f}')
        
        if i < len(positions):
            print()
    
    # Compare results
    print('\n' + '='*70)
    print('📊 MODEL COMPARISON RESULTS')
    print('='*70)
    print()
    
    print('Position Comparison (Random Forest vs XGBoost vs LightGBM):')
    print('-' * 80)
    print(f'{"Position":<8} | {"RF PR-AUC":<10} | {"XGB PR-AUC":<12} | {"LGB PR-AUC":<12} | {"Winner":<12} | {"Best PR-AUC"}')
    print('-' * 80)
    
    best_models = {}
    for position in positions:
        rf_ap = results['RandomForest'][position].get('test_ap', results['RandomForest'][position]['test_auc'])
        xgb_ap = results['XGBoost'][position].get('test_ap', results['XGBoost'][position]['test_auc'])
        lgb_ap = results['LightGBM'][position].get('test_ap', results['LightGBM'][position]['test_auc'])
        
        # Find the best model
        model_scores = {'RandomForest': rf_ap, 'XGBoost': xgb_ap, 'LightGBM': lgb_ap}
        best_model = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model]
        
        best_models[position] = best_model
        
        print(f'{position:<8} | {rf_ap:<10.3f} | {xgb_ap:<12.3f} | {lgb_ap:<12.3f} | {best_model:<12} | {best_score:.3f}')
    
    print()
    print('🎯 RECOMMENDED MODELS FOR BETTING (by PR-AUC):')
    print('-' * 40)
    for position in positions:
        best_model_type = best_models[position]
        if best_model_type != 'Tie':
            best_ap = results[best_model_type][position].get('test_ap', results[best_model_type][position]['test_auc'])
            print(f'{position:<8}: {best_model_type} (PR-AUC: {best_ap:.3f})')
        else:
            rf_ap = results['RandomForest'][position].get('test_ap', results['RandomForest'][position]['test_auc'])
            print(f'{position:<8}: Either model (PR-AUC: {rf_ap:.3f})')
    
    print()
    print('💡 INSIGHTS:')
    print('-' * 20)
    rf_wins = sum(1 for pos in best_models.values() if pos == 'RandomForest')
    xgb_wins = sum(1 for pos in best_models.values() if pos == 'XGBoost')
    lgb_wins = sum(1 for pos in best_models.values() if pos == 'LightGBM')
    
    print(f'Random Forest wins: {rf_wins}/{len(positions)} positions')
    print(f'XGBoost wins: {xgb_wins}/{len(positions)} positions')
    print(f'LightGBM wins: {lgb_wins}/{len(positions)} positions')
    
    if rf_wins > max(xgb_wins, lgb_wins):
        print('🌲 Random Forest is the overall better model by PR-AUC!')
    elif xgb_wins > max(rf_wins, lgb_wins):
        print('🚀 XGBoost is the overall better model by PR-AUC!')
    elif lgb_wins > max(rf_wins, xgb_wins):
        print('💡 LightGBM is the overall better model by PR-AUC!')
    else:
        print('🤝 Models perform similarly - consider ensemble approach!')
    
    return results


if __name__ == "__main__":
    # Generate the comprehensive training dataset
    datasets, combined_data = generate_training_data()
    
    # Train and compare both Random Forest and XGBoost models
    comparison_results = train_and_compare_all_models(tune_hyperparams=True)
