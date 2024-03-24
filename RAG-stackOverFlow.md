##  Using Vertex AI Vector Search and Vertex AI Embeddings for Text for StackOverflow Questions 

## Setup and requirements

# Task 1. Enable Vertex AI APIs
```

Click on the nevigantion menu icon on the top left of the console.

Select APIs & services > Enable APIs and services, and click on + Enable APIs and services.

Search for Vertex AI API, click on the first entry and then click on Enable, if it is not already enabled.

```

```Task 2. Open a Jupyter notebook in Vertex AI Workbench```

```
In your Google Cloud project, navigate to Vertex AI Workbench. In the top search bar of the Google Cloud console, enter Vertex AI Workbench, and click on the first result.
Use search to locate Vertex AI workbench
If you see a button at the top of the screen with the title Enable Notebooks API, click on it to enable the API.

Click on User managed notebooks and then click on Open JupyterLab for generative-ai-jupyterlab notebook.

The JupyterLab will run in a new tab.

Open Notebook action
On the Launcher, under Notebook, click on Python 3 (ipykernel) to open a new python notebook.

```

```Task 3. Set up the Jupyter notebook environment```

In the first cell, run the following command to install the Google Cloud Vertex AI, Cloud Storage and BigQuery SDKs. To run the command, execute SHIFT+ENTER

```
! pip3 install --upgrade google-cloud-aiplatform \
                        google-cloud-storage \
                        'google-cloud-bigquery[pandas]'
```

Restart kernel after installs so that your environment can access the new packages
```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```
Setup the environment values for your project.

```
PROJECT = !gcloud config get-value project
PROJECT_ID = PROJECT[0]
REGION = "us-west1"
```
Import and initialize the Vertex AI Python SDK.

```
import vertexai
vertexai.init(project = PROJECT_ID,
              location = REGION)
```

```Task 4. Prepare the data in BigQuery```

Import the libraries and initialize the BigQuery client.

```
import math
from typing import Any, Generator

import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID)
```
Define the BigQuery query for the remote dataset.

```
QUERY_TEMPLATE = """
        SELECT distinct q.id, q.title, q.body
        FROM (SELECT * FROM `bigquery-public-data.stackoverflow.posts_questions` where Score>0 ORDER BY View_Count desc) AS q
        LIMIT {limit} OFFSET {offset};
        """

```

Create a function to access the BigQuery data in chunks.

```
def query_bigquery_chunks(
    max_rows: int, rows_per_chunk: int, start_chunk: int = 0
) -> Generator[pd.DataFrame, Any, None]:
    for offset in range(start_chunk, max_rows, rows_per_chunk):
        query = QUERY_TEMPLATE.format(limit=rows_per_chunk, offset=offset)
        query_job = client.query(query)
        rows = query_job.result()
        df = rows.to_dataframe()
        df["title_with_body"] = df.title + "\n" + df.body
        yield df

```

Get a dataframe of 1000 rows for demonstration purposes.

```
df = next(query_bigquery_chunks(max_rows=1000, rows_per_chunk=1000))

# Examine the data
df.head()

```


```Task 5. Create text embeddings from BigQuery data```

Load the Vertex AI Embeddings for Text model.

```
from typing import List, Optional
from vertexai.preview.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

```

Define an embedding method that uses the model.

```
def encode_texts_to_embeddings(sentences: List[str]) -> List[Optional[List[float]]]:
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]

```

    According to the documentation, each request can handle up to 5 text instances. So we will need to split the BigQuery question results in batches of 5 before sending to the embedding API.

Create a generate_batches to split results in batches of 5 to be sent to the embeddings API.

```
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, List, Tuple

import numpy as np
from tqdm.auto import tqdm


# Generator function to yield batches of sentences
def generate_batches(
    sentences: List[str], batch_size: int
) -> Generator[List[str], None, None]:
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

```

Encapsulate the process of generating batches and calling the embeddings API in a method called encode_text_to_embedding_batched. This method also handles rate-limiting using time.sleep. For production use cases, you would want a more sophisticated rate-limiting mechanism that takes retries into account.

```
def encode_text_to_embedding_batched(
    sentences: List[str], api_calls_per_second: int = 10, batch_size: int = 5
) -> Tuple[List[bool], np.ndarray]:

    embeddings_list: List[List[float]] = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total=math.ceil(len(sentences) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())

    is_successful = [
        embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    ]
    embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in embeddings_list if embedding is not None])
    )
    return is_successful, embeddings_list_successful

```

Test the encoding function by encoding a subset of data and see if the embeddings and distance metrics make sense.

```
# Encode a subset of questions for validation
questions = df.title.tolist()[:500]
is_successful, question_embeddings = encode_text_to_embedding_batched(
    sentences=df.title.tolist()[:500]
)

# Filter for successfully embedded sentences
questions = np.array(questions)[is_successful]
```

Save the dimension size for later usage when creating the Vertex AI Vector Search index.

```
DIMENSIONS = len(question_embeddings[0])

print(DIMENSIONS)

```

Sort questions in order of similarity. According to the embedding documentation, the similarity of embeddings is calculated using the dot-product, with np.dot. Once you have the similarity score, sort the results and print them for inspection. 1 means very similar, 0 means very different.

```

import random

question_index = random.randint(0, 99)

print(f"Query question = {questions[question_index]}")

# Get similarity scores for each embedding by using dot-product.
scores = np.dot(question_embeddings[question_index], question_embeddings.T)

# Print top 20 matches
for index, (question, score) in enumerate(
    sorted(zip(questions, scores), key=lambda x: x[1], reverse=True)[:20]
):
    print(f"\t{index}: {question}: {score}")

```

Save the embeddings in JSONL format. The data must be formatted in JSONL format, which means each embedding dictionary is written as an individual JSON object on its own line.

```
import tempfile
from pathlib import Path

# Create temporary file to write embeddings to
embeddings_file_path = Path(tempfile.mkdtemp())

print(f"Embeddings directory: {embeddings_file_path}")

```

Write embeddings in batches to prevent out-of-memory errors. Notice we are only using 5000 questions so that the embedding creation process and indexing is faster. The dataset contains more than 50,000 questions. This step will take around 5 minutes.

```

import gc
import json

BQ_NUM_ROWS = 5000
BQ_CHUNK_SIZE = 1000
BQ_NUM_CHUNKS = math.ceil(BQ_NUM_ROWS / BQ_CHUNK_SIZE)

START_CHUNK = 0

# Create a rate limit of 300 requests per minute. Adjust this depending on your quota.
API_CALLS_PER_SECOND = 300 / 60
# According to the docs, each request can process 5 instances per request
ITEMS_PER_REQUEST = 5

# Loop through each generated dataframe, convert
for i, df in tqdm(
    enumerate(
        query_bigquery_chunks(
            max_rows=BQ_NUM_ROWS, rows_per_chunk=BQ_CHUNK_SIZE, start_chunk=START_CHUNK
        )
    ),
    total=BQ_NUM_CHUNKS - START_CHUNK,
    position=-1,
    desc="Chunk of rows from BigQuery",
):
    # Create a unique output file for each chunk
    chunk_path = embeddings_file_path.joinpath(
        f"{embeddings_file_path.stem}_{i+START_CHUNK}.json"
    )
    with open(chunk_path, "a") as f:
        id_chunk = df.id

        # Convert batch to embeddings
        is_successful, question_chunk_embeddings = encode_text_to_embedding_batched(
            sentences=df.title_with_body.to_list(),
            api_calls_per_second=API_CALLS_PER_SECOND,
            batch_size=ITEMS_PER_REQUEST,
        )

        # Append to file
        embeddings_formatted = [
            json.dumps(
                {
                    "id": str(id),
                    "embedding": [str(value) for value in embedding],
                }
            )
            + "\n"
            for id, embedding in zip(id_chunk[is_successful], question_chunk_embeddings)
        ]
        f.writelines(embeddings_formatted)

        # Delete the DataFrame and any other large data structures
        del df
        gc.collect()

```

```Task 6. Upload embeddings to Cloud Storage```

Upload the text-embeddings to Cloud Storage, so that Vertex AI Vector Search can access them later.

Define a bucket where you will store your embeddings.

```
BUCKET_URI = f"gs://{PROJECT_ID}-unique"

```

Create your Cloud Storage bucket.

```
! gsutil mb -l {REGION} -p {PROJECT_ID} {BUCKET_URI}

```

Upload the training data to a Google Cloud Storage bucket.

```
remote_folder = f"{BUCKET_URI}/{embeddings_file_path.stem}/"
! gsutil -m cp -r {embeddings_file_path}/* {remote_folder}

```

```Task 7. Create an Index in Vertex AI Vector Search for your embeddings```

Setup your index name and description.

```
DISPLAY_NAME = "stack_overflow"
DESCRIPTION = "question titles and bodies from stackoverflow"

```

Create the index. Notice that the index reads the embeddings from the Cloud Storage bucket. The indexing process can take from 45 minutes up to 60 minutes. Wait for completion, and then proceed. You can open a different Google Cloud Console page, navigate to Vertex AI Vector search, and see how the index is being created.
from google.cloud i

```

from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

DIMENSIONS = 768

tree_ah_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=DISPLAY_NAME,
    contents_delta_uri=remote_folder,
    dimensions=DIMENSIONS,
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    leaf_node_embedding_count=500,
    leaf_nodes_to_search_percent=80,
    description=DESCRIPTION,
)

```

Reference the index name to make sure it got created successfully.

```
INDEX_RESOURCE_NAME = tree_ah_index.resource_name
INDEX_RESOURCE_NAME

```

Using the resource name, you can retrieve an existing MatchingEngineIndex.

```

tree_ah_index = aiplatform.MatchingEngineIndex(index_name=INDEX_RESOURCE_NAME)

```

Create an IndexEndpoint so that it can be accessed via an API.

```

my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=DISPLAY_NAME,
    description=DISPLAY_NAME,
    public_endpoint_enabled=True,
)

```

Deploy your index to the created endpoint. This can take up to 15 minutes.

```

DEPLOYED_INDEX_ID = "deployed_index_id_unique"

DEPLOYED_INDEX_ID


my_index_endpoint = my_index_endpoint.deploy_index(
    index=tree_ah_index, deployed_index_id=DEPLOYED_INDEX_ID
)

my_index_endpoint.deployed_indexes

```

Verify number of declared items matches the number of embeddings. Each IndexEndpoint can have multiple indexes deployed to it. For each index, you can retrieve the number of deployed vectors using the index_endpoint._gca_resource.index_stats.vectors_count. The numbers may not match exactly due to potential rate-limiting failures incurred when using the embedding service.

```
number_of_vectors = sum(
    aiplatform.MatchingEngineIndex(
        deployed_index.index
    )._gca_resource.index_stats.vectors_count
    for deployed_index in my_index_endpoint.deployed_indexes
)

print(f"Expected: {BQ_NUM_ROWS}, Actual: {number_of_vectors}")

```

```Task 8. Create online queries```

After you build your indexes, you may query against the deployed index to find nearest neighbors.

Note: For the DOT_PRODUCT_DISTANCE distance type, the "distance" property returned with each MatchNeighbor actually refers to the similarity.

Create an embedding for a test question.

```

test_embeddings = encode_texts_to_embeddings(sentences=["Install GPU for Tensorflow"])

```

Test the query to retrieve the similar embeddings.

```
NUM_NEIGHBOURS = 10

response = my_index_endpoint.find_neighbors(
    deployed_index_id=DEPLOYED_INDEX_ID,
    queries=test_embeddings,
    num_neighbors=NUM_NEIGHBOURS,
)

response

```

Verify that the retrieved results are relevant by checking the StackOverflow links.

```
for match_index, neighbor in enumerate(response[0]):
    print(f"https://stackoverflow.com/questions/{neighbor.id}")

```
