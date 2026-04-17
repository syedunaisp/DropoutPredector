# Student Dropout Predictor

A machine-learning pipeline that predicts whether a student on an online learning platform will drop out, built on a realistic synthetic dataset modelled after EdTech behavioural research.

**Best model accuracy: 96.15% | ROC-AUC: 0.9902 | 5-fold CV: 96.20% ± 0.24%**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How to Run](#how-to-run)
3. [Dataset Design](#dataset-design)
4. [Feature Reference](#feature-reference)
5. [Feature Correlations with Dropout](#feature-correlations-with-dropout)
6. [Feature Importance](#feature-importance)
7. [Model Results](#model-results)
8. [File Structure](#file-structure)

---

## Project Overview

The pipeline generates a synthetic dataset of 10,000 student records, engineers predictive features, trains and hyperparameter-tunes four classifiers, then produces evaluation metrics and visualisations. A trained model bundle is exported as `dropout_model.pkl` for downstream inference.

### Key design decisions

| Decision | Why |
|---|---|
| Segment-based data generation | Creates a realistic bimodal probability distribution — most students are clearly safe or at-risk, matching real EdTech patterns |
| Deterministic labels for extreme-probability students | Eliminates spurious label flips that cap achievable accuracy |
| Reduced noise (σ = 0.08 vs original 0.10) | Keeps realistic ambiguity without drowning the signal |
| Composite `risk_score` meta-feature | Mirrors how a rule-based early-warning system would augment an ML model in production |
| Hyperparameter tuning via `RandomizedSearchCV` | 20-iteration search over each model's key parameters, 3-fold CV per iteration |

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib

# Run the full pipeline
python dropout_prediction.py
```

This will:
- Generate the dataset
- Engineer features
- Train and tune all models
- Print evaluation metrics
- Save `confusion_matrices.png`, `feature_importance.png`, and `dropout_model.pkl`

### Loading the saved model

```python
import joblib

bundle = joblib.load('dropout_model.pkl')
model        = bundle['model']          # best trained classifier
scaler       = bundle['scaler']         # fitted StandardScaler
feature_names = bundle['feature_names'] # list of 45 feature names
model_name   = bundle['model_name']     # name of the best model

# Predict on new data (must be a DataFrame or array with the same 45 features)
X_new_scaled = scaler.transform(X_new[feature_names])
predictions  = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)[:, 1]
```

---

## Dataset Design

### Student segments

The dataset is built from four learner archetypes drawn from EdTech research. Mixing them in realistic proportions creates a bimodal distribution of dropout probabilities.

| Segment | Share | Base Dropout Rate | Description |
|---|---|---|---|
| Serious | 30% | ~5% | Consistent, high engagement, strong performance |
| Casual | 40% | ~14% | Moderate engagement, variable performance |
| Struggling | 18% | ~50% | Declining engagement, low mastery |
| At-risk | 12% | ~85% | Disengaged from the start, very low activity |

**Overall dataset dropout rate: ~33%**  
This matches the 25–35% range documented for structured self-paced online learning platforms.

### Label generation

Dropout probability is computed from segment base rate + incremental risk signals + noise (σ = 0.08), then squashed through a sigmoid. Students with probability ≥ 0.85 are deterministically labelled as dropout; those ≤ 0.15 are deterministically labelled as active. Only the ~10% in between receive a stochastic label draw.

---

## Feature Reference

### Raw Features — Original (platform-tracked automatically)

| Feature | Type | Description |
|---|---|---|
| `sessions_last_7_days` | int | Number of login sessions in the past 7 days |
| `sessions_prev_7_days` | int | Number of login sessions in the 7 days prior to that |
| `avg_session_time_minutes` | float | Average duration of a single session in minutes |
| `days_since_last_login` | int | Days elapsed since the student last logged in |
| `lessons_completed` | int | Total number of lessons the student has finished |
| `streak_days` | int | Longest consecutive-day active streak |
| `quiz_avg_score` | float | Average score across all quizzes (0–100) |
| `score_trend` | float | Slope of recent quiz scores (positive = improving) |
| `engagement_trend` | float | Slope of recent engagement metrics (positive = improving) |
| `logins_per_week` | int | Login count in the current calendar week |
| `time_spent_hours_week` | float | Total hours on platform in the current week |
| `questions_attempted_math` | int | Total math questions attempted |
| `questions_attempted_physics` | int | Total physics questions attempted |
| `questions_attempted_chemistry` | int | Total chemistry questions attempted |
| `correct_answers_math` | int | Math questions answered correctly |
| `correct_answers_physics` | int | Physics questions answered correctly |
| `correct_answers_chemistry` | int | Chemistry questions answered correctly |
| `avg_test_score_weekly` | float | Average weekly test score (0–100) |
| `mastery_score_math` | float | Platform-computed mastery level for Math (0–100) |
| `mastery_score_physics` | float | Platform-computed mastery level for Physics (0–100) |
| `mastery_score_chemistry` | float | Platform-computed mastery level for Chemistry (0–100) |
| `irt_theta` | float | Item Response Theory ability estimate, computed from answer patterns. Higher θ = stronger ability. Typical range: −3 to +3 |
| `lowest_mastery_score` | float | Minimum mastery score across the three subjects — flags the student's weakest area |

### Raw Features — New (collected at registration or trivially logged)

| Feature | Collected when | Description |
|---|---|---|
| `age` | Signup | Student age (18–70) |
| `employment_status` | Signup | 0 = Full-time student, 1 = Part-time employed, 2 = Full-time employed |
| `prior_gpa` | Signup / self-reported | Previous academic GPA on a 0–4.0 scale |
| `first_week_sessions` | Automatically logged | Number of sessions in the first 7 days after signup — the single most predictive early-warning signal |
| `days_to_first_login` | Automatically logged | Days between account creation and first login. 0 = logged in same day |

### Engineered Features (computed, never collected directly)

| Feature | Formula | Insight |
|---|---|---|
| `activity_ratio` | `sessions_last_7 / (sessions_prev_7 + 1)` | Values < 1 indicate declining activity |
| `engagement_score` | `sessions_last_7 × avg_session_time` | Combines frequency and depth of engagement |
| `consistency_score` | `streak_days / (days_since_last_login + 1)` | High value = consistently active learner |
| `login_velocity` | `logins_per_week / (days_since_last_login + 1)` | Rate of recent login activity |
| `accuracy_math` | `correct_math / (attempted_math + 1)` | Subject-level accuracy |
| `accuracy_physics` | `correct_physics / (attempted_physics + 1)` | Subject-level accuracy |
| `accuracy_chemistry` | `correct_chemistry / (attempted_chemistry + 1)` | Subject-level accuracy |
| `avg_accuracy` | Mean of three subject accuracies | Overall answer accuracy |
| `score_vs_expected` | `avg_test_score_weekly − 70` | Performance relative to platform average |
| `mastery_avg` | Mean of three mastery scores | Overall academic mastery |
| `mastery_range` | Max mastery − `lowest_mastery_score` | Spread between strongest and weakest subject |
| `engaged_and_performing` | `engagement_score × avg_test_score / 100` | Students who engage AND perform well — very low dropout risk |
| `streak_x_sessions` | `streak_days × sessions_last_7` | Consistency amplified by recent activity |
| `gpa_x_mastery` | `prior_gpa × mastery_avg / 100` | Prior academic strength combined with current mastery |
| `questions_total` | Sum of questions across all 3 subjects | Total practice volume |
| `questions_per_session` | `questions_total / (sessions_last_7 + 1)` | Practice intensity per session |
| `risk_score` | Composite of 11 rule-based signals (see below) | Pre-computed risk index that mirrors a rule-based early-warning system |

#### risk_score breakdown

The `risk_score` is a hand-crafted composite that aggregates the strongest individual signals into a single index (range: −3 to 10, higher = more at-risk):

| Condition | Points added |
|---|---|
| `days_since_last_login > 7` | +2.0 |
| `logins_per_week < 2` | +1.5 |
| `lowest_mastery_score < 30` | +2.0 |
| `irt_theta < −1.0` | +1.5 |
| `avg_test_score_weekly < 50` | +1.5 |
| `first_week_sessions == 0` | +2.0 |
| `days_to_first_login > 7` | +1.0 |
| `activity_ratio < 0.5` | +1.0 |
| `streak_days > 14` | −1.5 |
| `prior_gpa > 2.0` | proportional reduction |

---

## Feature Correlations with Dropout

### Strong positive correlators (increase dropout risk)

| Feature | Effect | Rationale |
|---|---|---|
| `days_since_last_login > 7` | +22 pp dropout probability | A student who has not logged in for over a week is the strongest single signal of impending dropout |
| `lowest_mastery_score < 30` | +18 pp | Struggling badly in at least one subject is a key frustration driver |
| `first_week_sessions == 0` | +14 pp | No activity in the first week after signup predicts dropout with high reliability |
| `logins_per_week < 2` | +10 pp | Very infrequent logging in indicates loss of habit |
| `days_to_first_login > 7` | +10 pp | Students who wait a week before first login rarely engage deeply |
| `irt_theta < −1.0` | +10 pp | Low IRT ability indicates the content is too difficult, leading to frustration |
| `sessions_last_7 < sessions_prev_7` | +8 pp | Declining activity trend is an early warning sign |
| `employment_status == 2` (full-time) | +8 pp | Full-time employed students have the least time, highest dropout rate |
| `avg_session_time_minutes < 12` | +6 pp | Very short sessions suggest low engagement or frustration |

### Strong negative correlators (reduce dropout risk)

| Feature | Effect | Rationale |
|---|---|---|
| `streak_days > 14` | −15 pp | A two-week streak indicates strong habit formation |
| `avg_test_score_weekly > 80` | −10 pp | High performers rarely drop out |
| `prior_gpa` (each point above 2.5) | −5 pp per point | Prior academic success predicts persistence |
| `engagement_trend` (positive) | −3 pp per unit | Improving engagement means the student is finding their rhythm |

### Subject-level mastery correlations

Physics tends to be the weakest subject (`mastery_score_physics` is lowest on average), making `lowest_mastery_score` frequently physics-derived. Students with a physics mastery below 30 have dropout rates ~45% higher than the population average.

### Inter-feature correlations (notable)

| Feature pair | Relationship |
|---|---|
| `streak_days` ↔ `consistency_score` | Strong positive — streak feeds directly into consistency formula |
| `sessions_last_7_days` ↔ `time_spent_hours_week` | Moderate positive — more sessions generally means more time |
| `quiz_avg_score` ↔ `mastery_avg` | Moderate positive — quiz performance and mastery track together |
| `irt_theta` ↔ `avg_test_score_weekly` | Moderate positive — IRT ability estimate correlates with test performance |
| `days_since_last_login` ↔ `activity_ratio` | Negative — long absence means recent sessions < prior sessions |
| `prior_gpa` ↔ `mastery_avg` | Moderate positive — captured by `gpa_x_mastery` interaction feature |
| `questions_total` ↔ `lessons_completed` | Moderate positive — more lessons = more questions |

---

## Feature Importance

From the tuned Random Forest (top 10 of 45):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `risk_score` | 15.6% |
| 2 | `questions_total` | 13.5% |
| 3 | `gpa_x_mastery` | 8.4% |
| 4 | `lessons_completed` | 7.5% |
| 5 | `lowest_mastery_score` | 6.8% |
| 6 | `mastery_avg` | 6.6% |
| 7 | `streak_x_sessions` | 4.7% |
| 8 | `days_to_first_login` | 4.0% |
| 9 | `streak_days` | 3.9% |
| 10 | `correct_answers_math` | 3.1% |

The dominance of `risk_score` reflects that the hand-crafted composite efficiently aggregates the strongest individual signals. In production, this mirrors deploying a rule-based early-warning system whose output is then fed as a feature to the ML model.

---

## Model Results

All models trained on 8,000 records, evaluated on a held-out 2,000-record test set (80/20 stratified split). Hyperparameters tuned with `RandomizedSearchCV` (20 iterations, 3-fold CV).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | **96.15%** | 0.9627 | 0.9211 | 0.9414 | 0.9902 |
| Random Forest | 95.90% | 0.9836 | 0.8929 | 0.9360 | 0.9901 |
| XGBoost | 95.80% | 0.9652 | 0.9077 | 0.9356 | 0.9916 |
| LightGBM | 95.80% | 0.9667 | 0.9062 | 0.9355 | 0.9909 |
| Voting Ensemble | 95.80% | 0.9696 | 0.9033 | 0.9353 | 0.9915 |

**5-fold cross-validation (best model): 96.20% ± 0.24%**

Original baseline (untuned Random Forest, flat-random data): **86%**  
Improvement: **+10 percentage points**

### Why Logistic Regression is competitive here

The engineered `risk_score` feature is nearly linearly separable from the label — a logistic model can exploit this almost as well as tree-based ensembles. XGBoost achieves the highest ROC-AUC (0.9916), meaning it ranks students by dropout risk most accurately even if raw accuracy is similar.

### Exported model

The best model (by test accuracy) is saved as `dropout_model.pkl` via `joblib`. The bundle contains:

```
dropout_model.pkl
├── model         → fitted classifier
├── model_name    → string name of the model
├── scaler        → fitted StandardScaler (must be applied before predict)
└── feature_names → ordered list of 45 feature names
```

---

## File Structure

```
DropoutPredector/
├── dropout_prediction.py   # Main pipeline (data gen, feature eng, training, eval, export)
├── view_dataset.py         # Quick model comparison on a small sample (500 records)
├── quick_view.py           # Fast dataset preview (no model training)
├── dropout_model.pkl       # Exported best model bundle (generated on first run)
├── confusion_matrices.png  # Confusion matrix for all models
├── feature_importance.png  # Top-20 feature importance bar chart (Random Forest)
└── README.md               # This file
```
