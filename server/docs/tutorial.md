# Getting Started Tutorial: SimilaritySearchServer

This tutorial will guide you step-by-step through setting up the `SimilaritySearchServer`, creating your first Dataset, and executing asynchronous jobs using our Python SDK.

## Step 1: Starting the Server

From the project root, initialize the server in development mode by calling the Julia executable with the current project and passing the `serve` command:

```bash
julia --project=. -m SimilaritySearchServer serve
```

If it's the first time it is run, the system will automatically generate the default configuration and start listening at `http://127.0.0.1:8080`.

## Step 2: Preparing the Python Environment

In a separate terminal, ensure you have the Python dependencies installed:

```bash
cd python
pip install -r requirements.txt
```

Make sure to run your Python scripts from the `python/` folder so the local packages are recognized, or export the path to `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd)
```

## Step 3: Basic API Interaction

Create a short script or use an interactive Python console to test the connection:

```python
from similarity_search.client import SimilaritySearchClient

client = SimilaritySearchClient("http://127.0.0.1:8080")

# Check server health
print(client.readyz())
# Expected output: {'status': 'ok'}

# Create a new Dataset
ds_id = client.create_dataset()
print(f"My new dataset is: {ds_id}")
```

## Step 4: Running a Heavy Job (Job Spooling)

Suppose you want to process thousands of documents. Doing this by blocking an HTTP request is a bad idea, so we use the asynchronous Jobs API.

```python
from similarity_search.client import SimilaritySearchClient
from similarity_search.jobs import run_job

client = SimilaritySearchClient("http://127.0.0.1:8080")
ds_id = client.create_dataset()

# We define the heavy command arguments that the Julia Worker will execute
comando = ["--dataset", ds_id, "--batch-size", "1000"]

# We execute the job blocking only our Python thread while polling
print("Submitting batch processing...")
resultado = run_job(client, kind="searchbatch", command=comando)

print("Job Finished!")
print(resultado)
```

## Step 5: Stress Testing

To see the whole ecosystem working together, run our stress test script that orchestrates the entire process simulating real traffic:

```bash
cd python
python stress_client.py --jobs 3
```

The script will create a Dataset and submit multiple `searchbatch` jobs asynchronously, probing the server (via HTTP polling of `.job.toml` files) until the local server dispatches and processes all tasks.
