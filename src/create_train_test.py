import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "dataset/processed/phishing_clean.csv"
)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Result"],
    random_state=42
)

train_df.to_csv(
    "dataset/processed/train.csv",
    index=False
)

test_df.to_csv(
    "dataset/processed/test.csv",
    index=False
)

print(train_df.shape)
print(test_df.shape)