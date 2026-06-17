# federated/client.py

import flwr as fl
import numpy as np
import sys

from sklearn.metrics import (
    accuracy_score,
    log_loss
)

from model import (
    create_model,
    load_client_data
)

# -----------------------------------
# Client ID
# -----------------------------------

client_id = int(sys.argv[1])

# -----------------------------------
# Load Data
# -----------------------------------

X, y = load_client_data(client_id)

# -----------------------------------
# Create Model
# -----------------------------------

model = create_model()

# Initialize model once
model.fit(X, y)

# -----------------------------------
# Flower Client
# -----------------------------------

class FlowerClient(fl.client.NumPyClient):

    # Get model weights
    def get_parameters(self, config):
        return [
            model.coef_,
            model.intercept_
        ]

    # Set model weights
    def set_parameters(self, parameters):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

    # Local Training
    def fit(self, parameters, config):

        self.set_parameters(parameters)

        model.fit(X, y)

        # -----------------------------------
        # Differential Privacy
        # -----------------------------------

        coef = model.coef_.copy()
        intercept = model.intercept_.copy()

        noise_std = 0.01

        coef += np.random.normal(
            0,
            noise_std,
            coef.shape
        )

        intercept += np.random.normal(
            0,
            noise_std,
            intercept.shape
        )

        return (
            [coef, intercept],
            len(X),
            {}
        )

    # Local Evaluation
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        preds = model.predict(X)

        probs = model.predict_proba(X)

        accuracy = accuracy_score(
            y,
            preds
        )

        loss = log_loss(
            y,
            probs
        )

        return (
            loss,
            len(X),
            {
                "accuracy": accuracy
            }
        )


# -----------------------------------
# Start Client
# -----------------------------------

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)