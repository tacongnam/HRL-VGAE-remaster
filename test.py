
import json
from pathlib import Path

import numpy as np
import requests


JSON_PATH = Path("data/testcase/conus_rural_easy.json")
T_SCALE_FACTOR = 30
D_MAX_SCALE_FACTOR = 50
RANDOM_SEED = 42


def collect_r_records(data):
    requests = data.get("R", {})
    print(f"Number of R records: {len(requests)}")
    print("Type of R:", type(requests))

    if not isinstance(requests, list):
        raise ValueError('"R" must be a list.')

    return [
        request
        for request in requests
        if isinstance(request, dict)
    ]


def main():
    if not JSON_PATH.is_file():
        raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")

    with JSON_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    requests = data.get("R", [])

    if not isinstance(requests, list):
        raise ValueError('"R" must be a list.')

    records = [
        request for request in requests
        if isinstance(request, dict)
    ]

    t_values = [
        r["T"] for r in records
        if isinstance(r.get("T"), (int, float))
        and not isinstance(r.get("T"), bool)
        and r["T"] >= 0
    ]

    d_max_values = [
        r["d_max"] for r in records
        if isinstance(r.get("d_max"), (int, float))
        and not isinstance(r.get("d_max"), bool)
        and r["d_max"] >= 0
    ]

    if not t_values:
        raise ValueError('No valid numeric values found at ["R"][i]["T"].')

    if not d_max_values:
        raise ValueError('No valid numeric values found at ["R"][i]["d_max"].')

    # Estimate distribution parameters from original data.
    poisson_lambda = float(np.mean(t_values)) / T_SCALE_FACTOR
    exponential_scale = float(np.mean(d_max_values)) / D_MAX_SCALE_FACTOR

    if poisson_lambda <= 0 or exponential_scale <= 0:
        raise ValueError("Distribution parameters must be positive.")

    rng = np.random.default_rng(RANDOM_SEED)

    # Resample from the truncated Poisson distribution: 1 <= T <= 50.
    for r in records:
        if (
            isinstance(r.get("T"), (int, float))
            and not isinstance(r.get("T"), bool)
        ):
            while True:
                value = int(rng.poisson(poisson_lambda))
                if 1 <= value <= 50:
                    r["T"] = value
                    break

        # Resample from the truncated exponential distribution: 5 <= d_max <= 12.
        if (
            isinstance(r.get("d_max"), (int, float))
            and not isinstance(r.get("d_max"), bool)
        ):
            while True:
                value = float(rng.exponential(scale=exponential_scale))
                if 5 <= value <= 12:
                    r["d_max"] = value
                    break

    # Overwrite the original JSON file.
    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Updated file: {JSON_PATH}")
    print(f"Number of R records: {len(records)}")
    print(f"T ~ Truncated Poisson(lambda={poisson_lambda:.6g}), range [1, 50]")
    print(
        f"d_max ~ Truncated Exponential(scale={exponential_scale:.6g}), "
        "range [5, 12]"
    )
    print(f"Random seed: {RANDOM_SEED}")

if __name__ == "__main__":
    print("Starting script...")
    main()
    print("Finished successfully.")