import os
import sys

# Add the 'src' directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from admission_interface import AdmissionClient

if __name__ == "__main__":
    client = AdmissionClient()
    for round in range(5):  # Simulating 5 federated training rounds
        print(f"\n🔁 Round {round+1}")
        client.train(num_epochs=1)
        metrics = client.evaluate()
        print(f"✅ Evaluation metrics: {metrics}")

