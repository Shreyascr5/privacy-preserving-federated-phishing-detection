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
# Global Variables
# -----------------------------------

previous_accuracy = 0.0

# -----------------------------------
# Client ID
# -----------------------------------

client_id = int(sys.argv[1])

# -----------------------------------
# Load Client Data
# -----------------------------------

X, y = load_client_data(client_id)

# -----------------------------------
# Create Model
# -----------------------------------

model = create_model()

# Initialize model
model.fit(X, y)


# -----------------------------------
# Flower Client
# -----------------------------------

class FlowerClient(fl.client.NumPyClient):

    # -------------------
    # Get Parameters
    # -------------------
    def get_parameters(self, config):
        return [
            model.coef_,
            model.intercept_
        ]

    # -------------------
    # Set Parameters
    # -------------------
    def set_parameters(self, parameters):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

    # -------------------
    # Local Training
    # -------------------
    def fit(self, parameters, config):

        global previous_accuracy

        self.set_parameters(parameters)

        # Local training
        model.fit(X, y)

        # Local accuracy
        preds = model.predict(X)

        current_accuracy = accuracy_score(
            y,
            preds
        )

        # -------------------
        # Differential Privacy
        # -------------------

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

        # -------------------
        # Selective Updates
        # -------------------

        if current_accuracy > previous_accuracy:

            previous_accuracy = current_accuracy

            print(
                f"Client {client_id}: UPDATE SENT | Accuracy = {current_accuracy:.4f}"
            )

            sent = 1

        else:

            print(
                f"Client {client_id}: UPDATE SKIPPED | Accuracy = {current_accuracy:.4f}"
            )

            sent = 0

        return (
            [coef, intercept],
            len(X),
            {
                "accuracy": float(current_accuracy),
                "sent": sent
            }
        )

    # -------------------
    # Local Evaluation
    # -------------------
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
            float(loss),
            len(X),
            {
                "accuracy": float(accuracy)
            }
        )


# -----------------------------------
# Start Flower Client
# -----------------------------------

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)