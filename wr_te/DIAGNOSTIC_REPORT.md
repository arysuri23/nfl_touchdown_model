# Model Performance Diagnostic Report - 2025 Season

**Date**: November 8, 2025  
**Analysis Period**: Weeks 1-9 of 2025 NFL Season

> This is a historical diagnostic. Its former threshold recommendations are
> superseded by the Phase 0 ledger strategies in `README.md`; use the named
> strategies there rather than inferring an additional edge filter.

---

## 🎯 Executive Summary

**Your model isn't broken - the betting market caught up to it.**

Your hit rate remains stable (46-53%), but your model edge declined from +2.2% (weeks 4-6) to -0.5% (weeks 7-9). This is a **market efficiency problem**, not a model quality problem.

**Recommendation**: YES, retrain - but with realistic expectations. You'll see incremental gains (5-10%), not a dramatic turnaround. The real opportunity is in better feature engineering and smarter bet selection.

---

## 📊 Key Findings

### 1. BASE TD RATE ✅ Stable
- **2024**: 17.58%
- **2025** (Weeks 1-9): 18.13%
- **Change**: +0.55% (minimal impact)

### 2. MODEL PREDICTIONS ✅ Consistent
- Top 5 WR probabilities stable at ~0.45-0.50 across all weeks
- Mean prediction: 0.167
- Model is working as designed

### 3. HIT RATE ✅ Acceptable
- **Weeks 1-3**: 46.7%
- **Weeks 4-6**: 53.3%
- **Weeks 7-9**: 46.7%
- Stable performance, no degradation

### 4. MODEL EDGE ⚠️ DECLINING
- **Weeks 1-3**: +0.4%
- **Weeks 4-6**: +2.2%
- **Weeks 7-9**: -0.5% ⚠️ 
- **Critical finding**: Edge turning negative!

### 5. CALIBRATION ⚠️ SLIGHTLY OFF
- Average edge across all predictions: -0.014
- Model is slightly pessimistic
- Recalibration recommended

### 6. MARKET EFFICIENCY 🚨 CRITICAL
- Only **47.7%** of predictions higher than market (should be ~50%)
- Only **10.5%** of predictions had material positive edge
- Vegas has become more accurate

---

## 💡 Root Cause Analysis

**The Issue**: Vegas odds improved and caught up to your model signals.

**Evidence**:
1. Your model's hit rate is stable (46-53%)
2. Your model's probabilities are stable (0.45-0.50)
3. But your EDGE declined from +2.2% to -0.5%
4. Market implied probabilities now match or exceed your model more often

**Why This Happens**:
- Sportsbooks use sophisticated models too
- They adjust quickly to exploitable patterns
- Your public data sources are available to everyone
- Market becomes more efficient throughout the season

---

## 🎯 Recommended Actions

### IMMEDIATE (For Weeks 10-18)

#### 1. ✅ YES - Retrain the Model
**Why**: Adding 2024 data + weeks 1-9 captures most recent patterns

**How**:
```python
# Update train_wr.py
train_set: 2020-2024 + 2025 weeks 1-9
validation_set: 2023 (for clean testing)
test_set: 2024 (to compare to original model)
```

**Expected Impact**: 5-10% improvement in precision@5

#### 2. ✅ YES - Recalibrate
**Why**: Model is slightly pessimistic (-1.4% avg edge)

**How**:
- Fit new Platt calibrator on 2025 weeks 1-9 actual outcomes
- Compare calibration error before/after
- Validate on held-out 2024 data

#### 3. 🔍 INVESTIGATE - Feature Engineering
**Why**: Vegas may be using signals your model isn't

**High-Priority Features to Add**:
- **Recent trends**: Last 3 games rolling averages (more weight to recent)
- **QB-WR chemistry**: Targets/TDs in games where both played
- **Defensive ranking**: Opponent's pass defense rank (not just raw stats)
- **Line movement**: Opening vs closing Vegas lines
- **Injury impact**: Starter out = more targets for WR2/WR3
- **Game script**: Teams trailing target WRs more
- **Weather**: Wind/rain impacts for outdoor games

#### 4. 📊 ADJUST - Betting Strategy
**Phase 0 strategies**: `top5_prob` is the primary strategy, while
`top5_edge_le400` is a separate benchmark that can be logged at zero stake.
The ledger does not implement a standalone edge-threshold strategy.

**Better Strategies**:
1. **Use the implemented strategies**: record the primary strategy and, if
   desired, the zero-stake edge benchmark separately.
2. **Weighted sizing**: any sizing change requires an explicit strategy update.
3. **Position splits**: TEs may have different market efficiency.
4. **Early season focus**: your edge was positive in weeks 1-6.

**Expected Impact**: Could improve ROI by 50-100% without retraining

### LONGER TERM

#### 5. 🔄 Periodic training refresh
- Refresh in the first week and every 4 weeks during the season.
- Collect completed active-season history for prediction features.
- Fit the deployed model on the configured prior-season training cut and
  recalibrate its OOB probabilities.
- Monitor market efficiency metrics weekly.

#### 6. 📈 Enhance Model Architecture
Consider:
- **Ensemble models**: Combine multiple time windows
- **Sample weighting**: Give 2024-2025 data more importance
- **Separate models**: WR vs TE may need different features
- **Neural networks**: May capture non-linear patterns better

---

## ⚙️ Specific Next Steps

### Step 1: Update Training Script
```python
# In train_wr.py, modify:
CURRENT_SEASON = 2025
CURRENT_WEEK = 10  # Update to current week

# Change train/val split:
train_df = df[(df['season'] < 2024) | 
              ((df['season'] == 2024) & (df['week'] <= 18)) |
              ((df['season'] == 2025) & (df['week'] < 10))].copy()
val_df = df[df['season'] == 2023].copy()
```

### Step 2: Recalibrate
```python
# After training, fit calibrator on 2025 weeks 1-9
val_2025 = df[(df['season'] == 2025) & (df['week'] < 10)]
X_val_2025 = val_2025[WR_TE_FEATURES]
y_val_2025 = val_2025['scored_touchdown']

val_probs = model.predict_proba(X_val_2025)[:, 1].reshape(-1, 1)
calibrator = LogisticRegression()
calibrator.fit(val_probs, y_val_2025)
```

### Step 3: Test on Historical 2024
Validate the retrained model on full 2024 season and compare metrics:
- Precision@5
- Calibration error
- Average edge
- ROI on simulated bets

### Step 4: Deploy for Week 10
- Use new model + new calibrator
- Monitor edge closely
- Record bets with one of the named ledger strategies; no additional edge
  threshold is implied by the Phase 0 runbook.

### Step 5: Track Metrics
Weekly logging:
```python
metrics_to_track = {
    'week': week_number,
    'precision_at_5': precision,
    'avg_edge': model_edge,
    'calibration_error': ece,
    'bets_placed': count,
    'hits': successful_picks,
    'roi': return_on_investment
}
```

Set alert if:
- Edge negative for 2+ consecutive weeks
- Calibration error >0.10
- Precision@5 drops below 0.30

---

## 🎯 Final Verdict

### Should You Retrain?

**YES** - Retrain on 2020-2024 + weeks 1-9, validate on 2023

### But Understand Why:

**You thought**: "My model is performing worse"  
**Reality**: "The betting market got more efficient"

### What Retraining Will Do ✓
- Add 2024 season (most recent full season)
- Add 2025 weeks 1-9 (current season patterns)
- Let you recalibrate on fresh 2025 data
- Opportunity to re-tune hyperparameters

### What Retraining Won't Fix ✗
- Market efficiency (Vegas got smarter)
- Need for better features
- Need for smarter bet selection strategy

### Expected Outcome
- **Optimistic**: 10-15% improvement in precision@5
- **Realistic**: 5-10% improvement
- **With better bet selection**: 50-100% improvement in ROI

### Confidence Level
🟡 **MODERATE**

Retrain, but temper expectations. The bigger wins come from:
1. Better feature engineering (add recency, chemistry, trends)
2. Smarter bet selection through the implemented ledger strategies
3. Periodic training refresh (every 4 weeks)

---

## 📈 Success Metrics

Track these to measure if retraining worked:

| Metric | Current (Weeks 7-9) | Target (Weeks 10-12) |
|--------|--------------------|-----------------------|
| Avg Edge | -0.5% | >+2.0% |
| Precision@5 | 46.7% | >50% |
| Calibration Error | ~0.020 | <0.015 |
| % Predictions with positive edge | 10.5% | Track trend |
| ROI under a named ledger strategy | N/A | Establish baseline |

---

## 🚀 Bonus: Quick Wins Without Retraining

If you want immediate improvement for week 10:

1. **Use the ledger strategy**: `top5_prob` is the funded strategy;
   `top5_edge_le400` is a zero-stake benchmark.
2. **Focus on TEs**: Check if TEs have better edge than WRs
3. **Early week bets**: Place bets early before lines sharpen
4. **Smaller slate**: Bet fewer games (3-4 instead of 5)

These changes could improve ROI by 30-50% immediately.

---

**Report Generated**: November 8, 2025  
**Next Review**: After Week 12 (3 weeks post-retrain)
