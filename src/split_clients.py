import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "dataset/processed/phishing_clean.csv"
)

client1, temp = train_test_split(
    df,
    test_size=0.75,
    random_state=42
)

client2, temp = train_test_split(
    temp,
    test_size=2/3,
    random_state=42
)

client3, client4 = train_test_split(
    temp,
    test_size=0.5,
    random_state=42
)

Path(
    "dataset/clients"
).mkdir(
    parents=True,
    exist_ok=True
)

client1.to_csv(
    "dataset/clients/client1.csv",
    index=False
)

client2.to_csv(
    "dataset/clients/client2.csv",
    index=False
)

client3.to_csv(
    "dataset/clients/client3.csv",
    index=False
)

client4.to_csv(
    "dataset/clients/client4.csv",
    index=False
)

print("Client sizes:")

print(len(client1))
print(len(client2))
print(len(client3))
print(len(client4))

clients = [
    client1,
    client2,
    client3,
    client4
]

for i, client in enumerate(clients, start=1):
    print(
        f"\nClient {i}"
    )

    print(
        client["Result"].value_counts(
            normalize=True
        )
    )