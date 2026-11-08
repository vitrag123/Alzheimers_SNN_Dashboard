
import os
import json
import torch
import torch.nn as nn
import snntorch as snn
import joblib

from snntorch import surrogate


class AlzheimerSNN(nn.Module):

    def __init__(self, num_steps=20):
        super().__init__()

        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(171, 128)
        self.lif1 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(128, 64)
        self.lif2 = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

        self.fc_out = nn.Linear(64, 2)

    def forward(self, x):

        x = torch.clamp(x, -5, 5)
        x_prob = torch.sigmoid(x)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        output_sum = torch.zeros(
            x.shape[0], 2, device=x.device
        )

        for _ in range(self.num_steps):

            input_spikes = (
                torch.rand_like(x_prob) < x_prob
            ).float()

            spk1, mem1 = self.lif1(
                self.fc1(input_spikes), mem1
            )

            spk2, mem2 = self.lif2(
                self.fc2(spk1), mem2
            )

            output_sum += self.fc_out(spk2)

        return output_sum / self.num_steps


class SNNPredictor:

    def __init__(self, model_dir="models"):

        self.device = torch.device("cpu")

        model_path = os.path.join(
            model_dir, "alzheimer_snn_best.pth"
        )

        scaler_path = os.path.join(
            model_dir, "feature_scaler.pkl"
        )

        config_path = os.path.join(
            model_dir, "model_config.json"
        )

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.scaler = joblib.load(scaler_path)

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        self.model = AlzheimerSNN(
            num_steps=checkpoint["num_steps"]
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    def predict(self, features, mc_passes=10):

        """
        features: NumPy array of shape (n_windows, 171)
        Returns average class probabilities.
        """

        import numpy as np

        features = np.asarray(
            features, dtype=np.float32
        )

        if features.ndim != 2 or features.shape[1] != 171:
            raise ValueError(
                "Expected feature array of shape (n_windows, 171)"
            )

        if not np.isfinite(features).all():
            raise ValueError("Features contain invalid values.")

        # Apply the exact training scaler
        scaled = self.scaler.transform(features)

        x = torch.tensor(
            scaled, dtype=torch.float32
        ).to(self.device)

        probability_runs = []

        with torch.no_grad():

            for seed in range(mc_passes):

                torch.manual_seed(1000 + seed)

                logits = self.model(x)

                probabilities = torch.softmax(
                    logits, dim=1
                )

                probability_runs.append(
                    probabilities.cpu().numpy()
                )

        mean_probabilities = np.mean(
            probability_runs, axis=0
        )

        return {
            "healthy_probability": float(
                mean_probabilities[:, 0].mean()
            ),
            "alzheimer_probability": float(
                mean_probabilities[:, 1].mean()
            ),
            "window_probabilities": mean_probabilities
        }
