import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/accuracy_results.csv")

plt.figure(figsize=(8,5))

plt.bar(
    df["Method"],
    df["Accuracy"]
)

plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison of Federated Learning Approaches")

plt.tight_layout()

plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300
)

plt.show()