# federated/server.py

import flwr as fl

from strategy import strategy

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(
        num_rounds=5
    ),
    strategy=strategy
)