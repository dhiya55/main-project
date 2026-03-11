import os
import sys
from pathlib import Path

print(f"Python version: {sys.version}")
print(f"__file__: {__file__}")

modules_to_test = [
    'importlib._bootstrap',
    'importlib._bootstrap_external',
]

for mod_name in modules_to_test:
    print(f"\n--- Testing module: {mod_name} ---")
    try:
        mod = __import__(mod_name, fromlist=['*'])
        f = getattr(mod, '__file__', 'None')
        print(f"__file__: {f}")
        
        if f and f.startswith('<frozen'):
            print("Detected frozen module file path string.")
            
            print(f"os.path.abspath('{f}'):")
            try:
                print(f"  Result: {os.path.abspath(f)}")
            except Exception as e:
                print(f"  FAILED with {type(e).__name__}: {e}")
                
            print(f"os.path.exists('{f}'):")
            try:
                print(f"  Result: {os.path.exists(f)}")
            except Exception as e:
                print(f"  FAILED with {type(e).__name__}: {e}")
                
            print(f"Path('{f}').resolve():")
            try:
                print(f"  Result: {Path(f).resolve()}")
            except Exception as e:
                print(f"  FAILED with {type(e).__name__}: {e}")

            # This is often where WinError 123 happens
            full_path = os.path.join(os.getcwd(), f)
            print(f"os.path.exists(os.path.join(cwd, '{f}')):")
            try:
                print(f"  Result: {os.path.exists(full_path)}")
            except Exception as e:
                print(f"  FAILED with {type(e).__name__}: {e}")

    except Exception as e:
        print(f"Could not test {mod_name}: {e}")

print("\n--- System Path ---")
for p in sys.path:
    print(f"  {p}")
