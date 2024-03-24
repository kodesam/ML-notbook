####  Using Vertex AI Vector Search and Vertex AI Embeddings for Text for StackOverflow Questions ####


### Setup and requirements
```Task 1. Enable Vertex AI APIs```

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

```Task 5. Create text embeddings from BigQuery data```

```Task 6. Upload embeddings to Cloud Storage```

```Task 7. Create an Index in Vertex AI Vector Search for your embeddings```

```Task 8. Create online queries```
