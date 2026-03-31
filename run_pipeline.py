from src.data_prep import generate_synthetic_portfolio, split_portfolio
from src.expected_loss import run_el_pipeline

df = generate_synthetic_portfolio(n=50000)
df_train, df_test = split_portfolio(df)
results = run_el_pipeline(df_train, df_test)
print(f"\nEL Rate: {results['el_rate']:.2%}")