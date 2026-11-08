
from utils.snn_model import SNNPredictor
import numpy as np

predictor = SNNPredictor(model_dir="models")

# Temporary test input only.
# This is NOT real EEG data.
test_features = np.zeros((2, 171), dtype=np.float32)

result = predictor.predict(test_features)

print("Model loaded successfully!")
print("Healthy probability:", result["healthy_probability"])
print("Alzheimer probability:", result["alzheimer_probability"])
