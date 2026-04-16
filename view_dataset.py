from dropout_prediction import StudentDropoutPredictor
import pandas as pd

def analyze_best_model():
    predictor = StudentDropoutPredictor(n_samples=500) # Much smaller for near-instant results
    df = predictor.generate_data()
    
    print("\n" + "="*80)
    print("GENERATED DATASET PREVIEW (First 10 Students)")
    print("="*80)
    # Displaying a selection of columns to keep it readable
    cols_to_show = [
        'user_id', 'days_since_last_login', 'logins_per_week', 
        'avg_test_score_weekly', 'lowest_mastery_score', 'irt_theta', 'dropout'
    ]
    print(df[cols_to_show].head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("DATASET SUMMARY STATISTICS")
    print("="*80)
    print(df[cols_to_show].describe().to_string())
    
    print("\n" + "="*80)
    print("TRAINING AND COMPARING MODELS...")
    print("="*80)
    predictor.engineer_features()
    predictor.preprocess_and_split()
    predictor.train_models()
    results_df = predictor.evaluate_models()
    
    best_model_info = results_df.loc[results_df['F1 Score'].idxmax()]
    
    print("\n" + "="*80)
    print(f"BEST MODEL FOUND: {best_model_info['Model']}")
    print("="*80)
    print(f"Accuracy:  {best_model_info['Accuracy']:.4f}")
    print(f"Precision: {best_model_info['Precision']:.4f}")
    print(f"Recall:    {best_model_info['Recall']:.4f}")
    print(f"F1 Score:  {best_model_info['F1 Score']:.4f}")
    print("\nObjective: We use this model to flag students at high Risk of leaving (Dropout=1).")
    print("Recall is particularly important here so we don't miss students who might leave.")

if __name__ == "__main__":
    analyze_best_model()
