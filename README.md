# mlops-w-vertex-ai

This repo demonstrates how to build a pipeline that automatically trains a custom model either on a periodic schedule or when new data is inserted into the dataset using Vertex AI Pipelines and Cloud Run functions

### Setup instructions

<details>
  <summary>[1] pip installs</summary>

Run the following in a terminal:

```
pip3 install -r requirements.txt
```

</details>

<details>
  <summary>[2] Enable APIs and configure IAM</summary>

Replace values for `PROJECT_ID`, `PROJEC_NUM`, and `USER`, then run commands in terminal

[2.a] Set project and user login

```
gcloud config get-value project
export PROJECT_ID=

gcloud projects describe $PROJECT_ID --format="value(projectNumber)"
export PROJECT_NUM=

export USER=
```

[2.b] Grant roles in your Google Account

```
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/bigquery.admin
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/aiplatform.user
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/storage.admin
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/pubsub.editor
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/cloudfunctions.admin
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/logging.viewer
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/logging.configWriter
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/iam.serviceAccountUser
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/eventarc.admin
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/aiplatform.colabEnterpriseUser
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/artifactregistry.admin
gcloud projects add-iam-policy-binding $PROJECT_ID --member=user:$USER --role=roles/serviceusage.serviceUsageAdmin
```

[2.c] Enable GCP APIs

```
gcloud services enable artifactregistry.googleapis.com \
    bigquery.googleapis.com \
    cloudbuild.googleapis.com \
    cloudfunctions.googleapis.com \
    logging.googleapis.com \
    pubsub.googleapis.com \
    run.googleapis.com \
    storage-component.googleapis.com  \
    eventarc.googleapis.com \
    serviceusage.googleapis.com \
    aiplatform.googleapis.com
```

[2.d] Grant service account IAM

```
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:PROJECT_NUM-compute@developer.gserviceaccount.com" --role=roles/aiplatform.serviceAgent
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:PROJECT_NUM-compute@developer.gserviceaccount.com" --role=roles/eventarc.eventReceiver
```
</details>