import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/communication_results.csv"
)

plt.figure(figsize=(6,5))

plt.bar(
    df["Method"],
    df["Updates"]
)

plt.ylabel("Number of Updates")

plt.title(
    "Communication Cost Comparison"
)

plt.tight_layout()

plt.savefig(
    "results/communication_comparison.png",
    dpi=300
)

plt.show()