import sys
import os
sys.path.append(os.path.abspath('.'))

from model.model_1 import MLP

model = MLP(input_dim=15)
print("✅ Import and instantiation successful!")
