"""Single-process federated server/client simulation entry point.

This mode is intentionally reproducible for classroom demonstrations. It uses
Flower's FedAvg strategy to aggregate independently trained client updates.
"""
from federated.train_federated import main


if __name__ == "__main__":
    main()
