"""Colab entry point: trains the 3D ResNet bone age model.

The real pipeline lives in src/train.py; this stays as the familiar command.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.train import DEFAULT_DATA_PATTERNS, train

if __name__ == "__main__":
    train(data_patterns=DEFAULT_DATA_PATTERNS, output="final_knee_model_resnet34.pth",
          arch="resnet34", epochs=40, batch_size=4)
