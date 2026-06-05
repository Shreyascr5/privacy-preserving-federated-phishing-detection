import arff
import pandas as pd
from pathlib import Path


DATA_PATH="dataset/raw/Training Dataset.arff"


with open(DATA_PATH,"r") as file:
    dataset= arff.load(file)

columns=[attr[0] for attr in dataset["attributes"]]

df=pd.DataFrame(dataset["data"],columns=columns)


print("datashape shape:", df.shape)


for col in df.columns:
    df[col]= pd.to_numeric(df[col])


df["Result"]= df["Result"].replace({
    1:0,
    -1:1
})

print("\nClass Distribution:")
print(df["Result"].value_counts())


Path("dataset/preprocessd").mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    "dataset/processed/phishing_clean.csv",
    index=False
)

print("\nSAVED phishing_clean.csv")
