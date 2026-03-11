import os
import sys

# Add the project root to sys.path if not there (usually is when running from root)
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

try:
    from ML.predictor import predict_water
    print("Successfully imported ML.predictor")
except ImportError as e:
    print(f"Failed to import ML.predictor: {e}")
except Exception as e:
    print(f"Error during import: {e}")
    import traceback
    traceback.print_exc()
