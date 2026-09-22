"""Load and print the policy array saved alongside this script."""

from pathlib import Path

import numpy as np


def main():
    policy_path = Path(__file__).resolve().parent / "policy.npy"
    policy = np.load(policy_path)
    for row in policy:
        print(" [" + " ".join(f"{value:6.2f}" for value in row) + "]")


if __name__ == "__main__":
    main()
