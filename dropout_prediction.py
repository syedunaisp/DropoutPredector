import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Try to import XGBoost, fallback to Scikit-Learn's GradientBoosting if not available
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

class StudentDropoutPredictor:
    def __init__(self, n_samples=10000, random_seed=42):
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.df = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def generate_data(self):
        """Step 1 & 2: Generate Realistic Synthetic Dataset and Labels"""
        print(f"Generating {self.n_samples} synthetic student records...")
        sys.stdout.flush()
        
        data = {
            'user_id': np.arange(1, self.n_samples + 1),
            'sessions_last_7_days': np.random.poisson(lam=5, size=self.n_samples),
            'sessions_prev_7_days': np.random.poisson(lam=6, size=self.n_samples),
            'avg_session_time_minutes': np.random.gamma(shape=2, scale=10, size=self.n_samples) + 5,
            'days_since_last_login': np.random.exponential(scale=4, size=self.n_samples).astype(int),
            'lessons_completed': np.random.randint(0, 100, size=self.n_samples),
            'streak_days': np.random.zipf(a=2.0, size=self.n_samples) - 1,
            'quiz_avg_score': np.random.normal(loc=75, scale=12, size=self.n_samples),
            'score_trend': np.random.normal(loc=0, scale=5, size=self.n_samples),
            'engagement_trend': np.random.normal(loc=0, scale=2, size=self.n_samples),
            
            # New parameters requested by user
            'logins_per_week': np.random.poisson(lam=5, size=self.n_samples),
            'time_spent_hours_week': np.random.gamma(shape=2, scale=5, size=self.n_samples) + 2,
            'questions_attempted_math': np.random.poisson(lam=12, size=self.n_samples),
            'questions_attempted_physics': np.random.poisson(lam=10, size=self.n_samples),
            'questions_attempted_chemistry': np.random.poisson(lam=8, size=self.n_samples),
            'avg_test_score_weekly': np.random.normal(loc=70, scale=15, size=self.n_samples),
            'mastery_score_math': np.random.normal(loc=65, scale=20, size=self.n_samples),
            'mastery_score_physics': np.random.normal(loc=60, scale=20, size=self.n_samples),
            'mastery_score_chemistry': np.random.normal(loc=62, scale=20, size=self.n_samples),
            'irt_theta': np.random.normal(loc=0, scale=1, size=self.n_samples)
        }
        
        # Derived and Clipped New Parameters
        data['correct_answers_math'] = np.random.binomial(data['questions_attempted_math'], 0.75)
        data['correct_answers_physics'] = np.random.binomial(data['questions_attempted_physics'], 0.70)
        data['correct_answers_chemistry'] = np.random.binomial(data['questions_attempted_chemistry'], 0.72)
        
        # Clip values to realistic ranges
        data['quiz_avg_score'] = np.clip(data['quiz_avg_score'], 0, 100)
        data['streak_days'] = np.clip(data['streak_days'], 0, 100)
        data['days_since_last_login'] = np.clip(data['days_since_last_login'], 0, 60)
        
        data['avg_test_score_weekly'] = np.clip(data['avg_test_score_weekly'], 0, 100)
        data['mastery_score_math'] = np.clip(data['mastery_score_math'], 0, 100)
        data['mastery_score_physics'] = np.clip(data['mastery_score_physics'], 0, 100)
        data['mastery_score_chemistry'] = np.clip(data['mastery_score_chemistry'], 0, 100)
        
        self.df = pd.DataFrame(data)
        
        # Lowest mastery score across topics
        self.df['lowest_mastery_score'] = self.df[['mastery_score_math', 'mastery_score_physics', 'mastery_score_chemistry']].min(axis=1)

        
        # Step 2: Dropout Logic (Label Creation)
        # We calculate a dropout probability based on the rules provided
        # We add noise to make it realistic
        
        # Base probability
        prob = 0.15 * np.ones(self.n_samples)
        
        # Rule: days_since_last_login > 5 -> high probability
        prob += np.where(self.df['days_since_last_login'] > 5, 0.4, 0)
        
        # Rule: sessions_last_7_days < sessions_prev_7_days -> higher risk
        prob += np.where(self.df['sessions_last_7_days'] < self.df['sessions_prev_7_days'], 0.2, 0)
        
        # Rule: avg_session_time is very low -> higher risk
        prob += np.where(self.df['avg_session_time_minutes'] < 15, 0.15, 0)
        
        # Rule: streak_days is high -> lower risk
        prob -= np.where(self.df['streak_days'] > 10, 0.25, 0)
        
        # Rule: lowest mastery is very low -> high risk
        prob += np.where(self.df['lowest_mastery_score'] < 30, 0.3, 0)
        
        # Rule: IRT Theta is low -> high risk
        prob += np.where(self.df['irt_theta'] < -1.0, 0.2, 0)
        
        # Rule: Low login frequency -> higher risk
        prob += np.where(self.df['logins_per_week'] < 2, 0.2, 0)
        
        # Rule: High test scores -> lower risk
        prob -= np.where(self.df['avg_test_score_weekly'] > 85, 0.2, 0)
        
        # Engagement trend impact
        prob -= self.df['engagement_trend'] * 0.05
        
        # Add random noise
        prob += np.random.normal(0, 0.1, self.n_samples)
        
        # Sigmoid-like squashing and thresholding
        prob = 1 / (1 + np.exp(-10 * (prob - 0.5)))
        self.df['dropout'] = (np.random.random(self.n_samples) < prob).astype(int)
        
        print("Data generation complete.")
        return self.df

    def engineer_features(self):
        """Step 3: Feature Engineering"""
        print("Engineering new features...")
        # activity_ratio = sessions_last_7_days / (sessions_prev_7_days + 1)
        self.df['activity_ratio'] = self.df['sessions_last_7_days'] / (self.df['sessions_prev_7_days'] + 1)
        
        # engagement_score = sessions_last_7_days * avg_session_time_minutes
        self.df['engagement_score'] = self.df['sessions_last_7_days'] * self.df['avg_session_time_minutes']
        
        # consistency_score = streak_days / (days_since_last_login + 1)
        self.df['consistency_score'] = self.df['streak_days'] / (self.df['days_since_last_login'] + 1)
        
        # Accuracy per subject
        self.df['accuracy_math'] = self.df['correct_answers_math'] / (self.df['questions_attempted_math'] + 1)
        self.df['accuracy_physics'] = self.df['correct_answers_physics'] / (self.df['questions_attempted_physics'] + 1)
        self.df['accuracy_chemistry'] = self.df['correct_answers_chemistry'] / (self.df['questions_attempted_chemistry'] + 1)
        
        # Handle division by zero/Infs if any (already mitigated by +1 in denominator, but good practice)
        self.df.replace([np.inf, -np.inf], 0, inplace=True)
        self.df.fillna(0, inplace=True)
        
        print("Feature engineering complete.")
        return self.df

    def preprocess_and_split(self):
        """Step 4: Data Preprocessing and Splitting"""
        print("Preprocessing data and splitting into train/test sets...")
        
        # Features to use for modeling (excluding user_id and label)
        feature_cols = [col for col in self.df.columns if col not in ['user_id', 'dropout']]
        X = self.df[feature_cols]
        y = self.df['dropout']
        
        # Split 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_seed, stratify=y)
        
        # Scaling features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store for later
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        self.feature_names = feature_cols
        
        print(f"X_train shape: {self.X_train.shape}, X_test shape: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """Step 5: Train Models (Logistic Regression, Random Forest, XGBoost/GBM)"""
        print("Training models...")
        
        # 1. Logistic Regression
        lr = LogisticRegression(random_state=self.random_seed, max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_seed, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # 3. Gradient Boosting (XGBoost or fallback)
        if HAS_XGB:
            print("Using XGBoost Classifier...")
            gbm = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_seed, use_label_encoder=False, eval_metric='logloss')
        else:
            print("XGBoost not found. Falling back to Scikit-Learn GradientBoostingClassifier...")
            gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_seed)
        
        gbm.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gbm
        
        print("Modeling training phase complete.")

    def evaluate_models(self):
        """Step 6: Evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred)
            }
            results.append(metrics)
            
            print(f"\n[{name}]")
            print(f"Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall:    {metrics['Recall']:.4f}")
            print(f"F1 Score:  {metrics['F1 Score']:.4f}")
            
        return pd.DataFrame(results)

    def plot_results(self):
        """Step 8: Plotting and Visualization"""
        # Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f"Confusion Matrix: {name}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        print("\nConfusion matrices saved as 'confusion_matrices.png'")
        
        # Feature Importance (for Random Forest)
        plt.figure(figsize=(10, 6))
        rf_model = self.models['Random Forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        sns.barplot(x=importances[indices], y=np.array(self.feature_names)[indices], palette='viridis')
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Absolute Importance")
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved as 'feature_importance.png'")
        
        # Feature Importance printout
        print("\n" + "-"*30)
        print("Top Contributors to Dropout Predictability (Random Forest):")
        for i in range(10):
            print(f"{i+1}. {self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        print("-"*30)

    def run_pipeline(self):
        """Execute the full workflow"""
        df = self.generate_data()
        
        print("\nSample Dataset (First 5 rows):")
        print(df.head())
        
        self.engineer_features()
        self.preprocess_and_split()
        self.train_models()
        self.evaluate_models()
        self.plot_results()

if __name__ == "__main__":
    predictor = StudentDropoutPredictor()
    predictor.run_pipeline()
