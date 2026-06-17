import pandas as pd
from pathlib import Path

df = pd.read_csv(
    "dataset/processed/train.csv"
)

phishing = df[df["Result"] == 1]
legitimate = df[df["Result"] == 0]

# Client 1
c1_phish = phishing.sample(
    n=1500,
    random_state=42
)

c1_legit = legitimate.sample(
    n=375,
    random_state=42
)

client1 = pd.concat(
    [c1_phish, c1_legit]
)

# Remove used rows
phishing = phishing.drop(c1_phish.index)
legitimate = legitimate.drop(c1_legit.index)

# Client 2
c2_phish = phishing.sample(
    n=375,
    random_state=42
)

c2_legit = legitimate.sample(
    n=1500,
    random_state=42
)

client2 = pd.concat(
    [c2_phish, c2_legit]
)

phishing = phishing.drop(c2_phish.index)
legitimate = legitimate.drop(c2_legit.index)

# Client 3
c3_phish = phishing.sample(
    n=800,
    random_state=42
)

c3_legit = legitimate.sample(
    n=800,
    random_state=42
)

client3 = pd.concat(
    [c3_phish, c3_legit]
)

phishing = phishing.drop(c3_phish.index)
legitimate = legitimate.drop(c3_legit.index)

# Client 4
client4 = pd.concat(
    [phishing, legitimate]
)

Path(
    "dataset/non_iid_clients"
).mkdir(
    exist_ok=True
)

client1.to_csv(
    "dataset/non_iid_clients/client1.csv",
    index=False
)

client2.to_csv(
    "dataset/non_iid_clients/client2.csv",
    index=False
)

client3.to_csv(
    "dataset/non_iid_clients/client3.csv",
    index=False
)

client4.to_csv(
    "dataset/non_iid_clients/client4.csv",
    index=False
)

for idx, client in enumerate(
    [client1, client2, client3, client4],
    start=1
):
    print(
        f"\nClient {idx}"
    )

    print(
        client["Result"]
        .value_counts(normalize=True)
        * 100
    )