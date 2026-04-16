import pandas as pd
import numpy as np

def generate_sample_data(n_samples=10):
    data = {
        'user_id': np.arange(1, n_samples + 1),
        'logins_per_week': np.random.poisson(lam=5, size=n_samples),
        'time_spent_hours_week': np.random.gamma(shape=2, scale=5, size=n_samples) + 2,
        'avg_test_score_weekly': np.random.normal(loc=70, scale=15, size=n_samples),
        'lowest_mastery_score': np.random.normal(loc=55, scale=20, size=n_samples),
        'irt_theta': np.random.normal(loc=0, scale=1, size=n_samples),
        'days_since_last_login': np.random.randint(0, 30, size=n_samples),
        'dropout': np.random.randint(0, 2, size=n_samples)
    }
    df = pd.DataFrame(data)
    print(df.to_string(index=False))

if __name__ == "__main__":
    generate_sample_data()
