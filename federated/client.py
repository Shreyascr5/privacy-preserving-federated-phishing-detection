# federated/client.py

import flwr as fl

from model import (
    create_model,
    load_client_data
)

import numpy as np
import sys

client_id = int(sys.argv[1])

X, y = load_client_data(
    client_id
)

model = create_model()

# initialize model
model.fit(
    X[:10],
    y[:10]
)


class FlowerClient(
    fl.client.NumPyClient
):

    def get_parameters(
        self,
        config
    ):
        return [
            model.coef_,
            model.intercept_
        ]

    def set_parameters(
        self,
        parameters
    ):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

    def fit(
        self,
        parameters,
        config
    ):

        self.set_parameters(
            parameters
        )

        model.fit(
            X,
            y
        )

        return (
            self.get_parameters(config),
            len(X),
            {}
        )

    def evaluate(
        self,
        parameters,
        config
    ):

        self.set_parameters(
            parameters
        )

        loss = 0.0

        accuracy = model.score(
            X,
            y
        )

        return (
            loss,
            len(X),
            {
                "accuracy": accuracy
            }
        )


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)