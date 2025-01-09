import flwr as fl
import numpy as np
import pickle

# Server configuration
config = fl.server.ServerConfig(
    num_rounds=5,
    round_timeout=600,  # Timeout for each round (10 minutes)
)

# Define the aggregation strategy
strategy = fl.server.strategy.FedAvg(
    fit_metrics_aggregation_fn=lambda metrics: {
        "accuracy": np.sum(
            [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
        ) / np.sum([num_examples for num_examples, _ in metrics]),
    },
    evaluate_metrics_aggregation_fn=lambda metrics: {
        "accuracy": np.sum(
            [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
        ) / np.sum([num_examples for num_examples, _ in metrics]),
    },
)

# Start the server with the specified configuration and strategy
print("Starting the Flower server...")
history = fl.server.start_server(
    server_address="localhost:8080",  # Change this if running on a different host
    config=config,
    strategy=strategy,
)
print("Flower server training completed!")

# Check the returned history object to ensure weights are present
if history and "parameters" in history.metrics_distributed:
    final_weights = history.metrics_distributed["parameters"]
    print("Final model weights retrieved.")

    # Save the final weights to a file (optional)
    import pickle
    with open("final_model_weights.pkl", "wb") as f:
        pickle.dump(final_weights, f)
else:
    print("No weights found in the final metrics.")
