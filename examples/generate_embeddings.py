import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to Python path so we can import our module
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd  # Not installed by default, so you need to install it with `pip install pandas`
from dotenv import find_dotenv, load_dotenv

from api_request_parallel_processor import process_api_requests_from_file

load_dotenv(find_dotenv())

MODEL_NAME = "voyage-3-large"
# Use paths relative to this script's directory
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "example_requests_to_parallel_process.jsonl"
OUTPUT_FILE = SCRIPT_DIR / "results.jsonl"
PARQUET_FILE = SCRIPT_DIR / "results.parquet"

# Generate 10,000 dummy requests to test the parallel processing script
n_requests = 10_000
jobs = [
    {
        "input": f"This is a test sentence {x}",
        "model": MODEL_NAME,
        "input_type": "document",  # "document" or "query"
        "metadata": {"id": x},
    }
    for x in range(n_requests)
]
with open(INPUT_FILE, "w") as f:
    for job in jobs:
        json_string = json.dumps(job)
        f.write(json_string + "\n")

print(f"Generated {n_requests} requests in {INPUT_FILE}")

# Process the requests in parallel
asyncio.run(
    process_api_requests_from_file(
        requests_filepath=str(INPUT_FILE),
        save_filepath=str(OUTPUT_FILE),
        request_url="https://api.voyageai.com/v1/embeddings",
        api_key=os.getenv("VOYAGE_API_KEY"),
        max_requests_per_minute=2_000 * 0.8,  # 80% of maximum 2,000 requests per minute
        max_tokens_per_minute=3_000_000
        * 0.8,  # 80% of maximum 3,000,000 tokens per minute
        conservative_factor=0.2,  # Voyage AI is sensitive to bursting, so start with much lower capacity
        model_name=MODEL_NAME,
        max_attempts=8,
        logging_level=logging.INFO,
    )
)

# Load, check and save the results
results = []
failed_requests = 0
with open(OUTPUT_FILE, "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            text = data[0]["input"]
            input_type = data[0]["input_type"]
            embedding = data[1]["data"][0]["embedding"]
            id = data[2]["id"]
            results.append(
                {
                    "id": id,
                    "text": text,
                    "input_type": input_type,
                    "embedding": embedding,
                }
            )
        except Exception as e:
            failed_requests += 1
            print(f"Error: {e}")
            continue

print(f"Number of successful results: {len(results)}")
print(f"Number of failed requests: {failed_requests}")
print(f"Saving results to {PARQUET_FILE}")
pd.DataFrame(results).to_parquet(PARQUET_FILE, index=False)
