import os

# Get the current file's directory
current_dir = os.path.dirname(__file__)

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

print(parent_dir)

print("\nokay\n")
print(os.pardir)