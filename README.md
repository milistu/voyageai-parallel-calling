# Voyage AI Parallel API Processor

**🚀 Generate 10,000 embeddings in minutes instead of hours** - This tool processes massive embedding datasets 10x faster while staying within Voyage AI's rate limits and optimizing your costs.

![Voyage AI Logo](https://www.voyageai.com/logo-v2.svg)

## Why This Matters ⚡

Processing embeddings one-by-one is like taking the bus when you could fly. If you're working with thousands of texts for RAG, semantic search, or ML pipelines, you **need** parallel processing. This isn't just faster - it's the difference between waiting hours vs. minutes.

**The numbers speak for themselves:**
- ✅ **10x faster** than sequential processing
- ✅ **Rate limit optimized** - no failed requests
- ✅ **Cost efficient** - maximize your API quota
- ✅ **Production ready** - handles errors, retries, and logging

## Key Features

* **🔄 Parallel Processing**: Concurrent API requests that maximize throughput
* **⚡ Smart Rate Limiting**: Adaptive algorithm designed specifically for Voyage AI's limits
* **🎯 Burst Protection**: Conservative startup prevents rate limit violations
* **🔧 Intelligent Tokenization**: Uses official Voyage AI tokenizers from HuggingFace
* **💾 Memory Efficient**: Streams requests from files to handle massive datasets
* **🛡️ Error Resilient**: Automatic retries with exponential backoff
* **📊 Progress Tracking**: Real-time logging and status updates

## Quick Start

### Installation

This project requires **Python 3.9+**.

1. **Clone and setup:**
```bash
git clone https://github.com/your-username/voyageai-parallel-calling.git
cd voyageai-parallel-calling
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Set your API key:**
Create a `.env` file:
```bash
VOYAGE_API_KEY=your-voyage-api-key-here
```

> 🔑 **Get your API key**: [Voyage AI Console](https://docs.voyageai.com/docs/api-key)

### Basic Usage

**Generate embeddings for 10,000 texts in under 5 minutes:**

```bash
python examples/generate_embeddings.py
```

That's it! The script will:
1. Generate 10,000 sample requests
2. Process them in parallel via Voyage AI
3. Save results as both JSONL and Parquet formats
4. Show you success/failure statistics

### Custom Processing

```bash
python api_request_parallel_processor.py \
  --requests_filepath your_requests.jsonl \
  --save_filepath results.jsonl \
  --max_requests_per_minute 1600 \
  --max_tokens_per_minute 2400000 \
  --model_name voyage-3-large
```

## Input File Format

Create a JSONL file where each line represents one embedding request:

```json
{"input": "Your text to embed", "model": "voyage-3-large", "input_type": "document", "metadata": {"id": 1}}
{"input": "Another text to embed", "model": "voyage-3-large", "input_type": "query", "metadata": {"id": 2}}
```

**Fields explained:**
- `input`: The text to embed (required)
- `model`: Voyage AI model name (required)
- `input_type`: "document" for passages, "query" for search queries (optional)
- `metadata`: Any additional data you want preserved (optional)

## Generating Request Files

### Simple Text List
```python
import json

texts = ["Text 1", "Text 2", "Text 3"]  # Your texts here
filename = "requests.jsonl"

jobs = [
    {
        "input": text,
        "model": "voyage-3-large", 
        "input_type": "document",
        "metadata": {"id": i}
    }
    for i, text in enumerate(texts)
]

with open(filename, "w") as f:
    for job in jobs:
        f.write(json.dumps(job) + "\n")
```

### From CSV/DataFrame
```python
import pandas as pd
import json

df = pd.read_csv("your_data.csv")
filename = "requests.jsonl"

with open(filename, "w") as f:
    for _, row in df.iterrows():
        request = {
            "input": row["text_column"],
            "model": "voyage-3-large",
            "input_type": "document",
            "metadata": {"id": row["id"], "category": row["category"]}
        }
        f.write(json.dumps(request) + "\n")
```

### Processing Results
```python
import json
import pandas as pd

# Load results
results = []
with open("results.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        request_data = data[0]  # Original request
        response_data = data[1]  # API response  
        metadata = data[2] if len(data) > 2 else {}  # Your metadata
        
        results.append({
            "id": metadata.get("id"),
            "text": request_data["input"],
            "embedding": response_data["data"][0]["embedding"]
        })

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(f"Generated {len(df)} embeddings")

# Save in your preferred format
df.to_parquet("embeddings.parquet")  # Efficient storage
df.to_csv("embeddings.csv")  # Human readable
```

### Handling Errors
```python
# Check for failed requests in results
failed_requests = []
with open("results.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if len(data) == 3 and isinstance(data[1], list):  # Error format
            failed_requests.append(data)

print(f"Failed requests: {len(failed_requests)}")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `requests_filepath` | Required | Path to your JSONL input file |
| `save_filepath` | Auto-generated | Where to save results |
| `request_url` | `https://api.voyageai.com/v1/embeddings` | Voyage AI API endpoint |
| `api_key` | From `.env` | Your Voyage API key |
| `max_requests_per_minute` | 1000 | Conservative rate limit (adjust based on your tier) |
| `max_tokens_per_minute` | 1500000 | Conservative token limit |
| `conservative_factor` | 0.1 | Factor to multiply the `max_requests_per_minute` and `max_tokens_per_minute` by to start with. Voyage AI is sensitive to bursting, so start with much lower capacity. |
| `model_name` | `voyage-3-large` | Voyage AI model for tokenization |
| `max_attempts` | 5 | Retry attempts for failed requests |
| `logging_level` | `INFO` | Logging verbosity |

### Rate Limits by Tier 📊

Check your limits at [Voyage AI Console](https://docs.voyageai.com/docs/rate-limits):
> ⚠️ **Always use maximum 80% of your limits** to avoid rate limit errors

## Important Notes ⚠️

### Voyage AI Specifics

1. **Burst Sensitivity**: Voyage AI has strict burst protection. This tool starts conservatively and ramps up gradually.

2. **Model Support**: Works with all Voyage AI embedding models.

3. **Token Counting**: Uses official Voyage AI tokenizers from HuggingFace for accurate token estimation.

### Performance Tips

- **Start small**: Test with 100 requests first
- **Monitor logs**: Watch for rate limit warnings
- **Adjust conservatively**: Lower rates if you see failures
- **Use SSDs**: Faster disk I/O helps with large files

### Common Issues

**Rate Limit Errors:**
Reduce `max_requests_per_minute` and `max_tokens_per_minute` by 20% and try again.

**Memory Issues:**
The script will run without any issue on large files (1M+ requests). However, when you want to process and save results in memory, you may run into memory issues. In that case, you can split the file into smaller chunks and process them one by one.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Project Origin

Adapted from the [OpenAI API Request Parallel Processor](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py) and inspired by the [Anthropic Parallel Processor](https://github.com/milistu/anthropic-parallel-calling).
