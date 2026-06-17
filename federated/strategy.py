import flwr as fl


def weighted_average(metrics):

    accuracies = [
        num_examples * m["accuracy"]
        for num_examples, m in metrics
    ]

    examples = [
        num_examples
        for num_examples, _ in metrics
    ]

    return {
        "accuracy": sum(accuracies) / sum(examples)
    }


def fit_metrics_aggregation(metrics):

    total_sent = sum(
        m.get("sent", 0)
        for _, m in metrics
    )

    return {
        "updates_sent": total_sent
    }


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4,
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=fit_metrics_aggregation
)