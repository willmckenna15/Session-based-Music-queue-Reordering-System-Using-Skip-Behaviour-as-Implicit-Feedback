import pandas as pd
df = pd.read_parquet('../RAW Data/training_data.parquet')
print(df['actively_selected'].value_counts(normalize=True))