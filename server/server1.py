import flwr as fl
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters_aggregated is not None:
            self.parameters = parameters_aggregated.tensors
        return parameters_aggregated, metrics_aggregated

strategy = SaveModelStrategy(
    fit_metrics_aggregation_fn=lambda metrics: {
        "accuracy": np.mean([m.get("accuracy", 0.0) for _, m in metrics])
    },
    evaluate_metrics_aggregation_fn=lambda metrics: {
        "accuracy": np.mean([m.get("accuracy", 0.0) for _, m in metrics])
    },
)

print("Starting the Flower server...")
history = fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)

if hasattr(strategy, 'parameters'):
    final_weights = [fl.common.parameter.bytes_to_ndarray(tensor) for tensor in strategy.parameters]
    
    model = Sequential([
        Dense(512, activation="relu", input_shape=(4096,)),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.set_weights(final_weights)
    model.save("final_model.h5")
    print("Model saved successfully.")
else:
    print("Training completed but no parameters were captured.")