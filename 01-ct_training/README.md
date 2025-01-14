# continuous training pipeline

> This folder demonstrates how to build a pipeline that trains a custom model either on a periodic schedule or when new data is inserted into the dataset using Vertex AI Pipelines and Cloud Run functions

**high-level objectives**

1. Prepare dataset in BigQuery
2. Create and upload a custom training package for Vertex Training
3. Build a Vertex AI Pipeline that handles continuous training and deployment
4. Run pipeline manually (and optionally via schedule)
5. Create a **pipeline trigger** using `Cloud Function with an Eventarc` that runs the pipeline when new data is inserted into the BigQuery dataset

## custom training package

We'll create a Python package that contains the code for training a custom model in Vertex AI with a [prebuilt container](https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container). This package will run as one of the steps in our continuous training pipeline

the structure of our package should look like this:

```
training_package
├── __init__.py
├── setup.py
└── trainer
    ├── __init__.py
    └── task.py
```

## custom vertex pipeline

**pipeline steps**

1. execute the custom training package
2. upload the trained model to the Vertex AI Model Registry 
3. run a model evaluation job
4. configure email notifications for pipeline completion (success or failure)

<details>
  <summary>Pipeline DAG in Google Cloud console</summary>

<img src='../imgs/ct_pipeline_v1.png' width='672' height='1085'>
    
</details>


<details>
  <summary>pipeline code</summary>

```
    # Notification task
    notify_task = VertexNotificationEmailOp(
        recipients= EMAIL_RECIPIENTS
    )
    
    with dsl.ExitHandler(notify_task, name='MLOps Continuous Training Pipeline'):
        # Train the model
        custom_job_task = (
            CustomTrainingJobOp(
                project=project,
                display_name=training_job_display_name,
                worker_pool_specs=worker_pool_specs,
                base_output_directory=base_output_dir,
                location=location
            )
        ).set_display_name("custom-train")

        # Import the unmanaged model
        import_unmanaged_model_task = (
            importer(
                artifact_uri=artifacts_dir,
                artifact_class=artifact_types.UnmanagedContainerModel,
                metadata={
                    "containerSpec": {
                        "imageUri": prediction_container_uri,
                    },
                },
            )
            .set_display_name("import-trained-model")
            .after(custom_job_task)
        )

        with dsl.If(existing_model == True):
            # Import the parent model to upload as a version
            import_registry_model_task = (
                importer(
                    artifact_uri=parent_model_artifact_uri,
                    artifact_class=artifact_types.VertexModel,
                    metadata={
                        "resourceName": parent_model_resource_name
                    },
                )
                .set_display_name("import-existing-model")
                .after(import_unmanaged_model_task)
            )
            
            # Upload the model as a version
            model_version_upload_op = ModelUploadOp(
                project=project,
                location=location,
                display_name=model_display_name,
                parent_model=import_registry_model_task.outputs["artifact"],
                unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
            )

        with dsl.Else():
            # Upload the model
            model_upload_op = (
                ModelUploadOp(
                    project=project,
                    location=location,
                    display_name=model_display_name,
                    unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
                )
                .set_display_name("upload-new-model")
            )
        
        # Get the model (or model version)
        model_resource = OneOf(
                model_version_upload_op.outputs["model"], 
                model_upload_op.outputs["model"]
        )

        # Batch prediction
        batch_predict_task = (
            ModelBatchPredictOp(
                project=project,
                job_display_name=batch_prediction_job_display_name,
                model=model_resource,
                location=location,
                instances_format=batch_predictions_input_format,
                predictions_format=batch_predictions_output_format,
                gcs_source_uris=test_data_gcs_uri,
                gcs_destination_output_uri_prefix=batch_predictions_gcs_prefix,
                machine_type='n1-standard-4'
            )
            .set_display_name("batch-prediction")
        )
        
        # Evaluation task
        evaluation_task = (
            ModelEvaluationRegressionOp(
                project=project,
                target_field_name=target_field_name,
                location=location,
                model=model_resource,
                predictions_format=batch_predictions_output_format,
                predictions_gcs_source=batch_predict_task.outputs["gcs_output_directory"],
                ground_truth_format=ground_truth_format,
                ground_truth_gcs_source=ground_truth_gcs_source
            )
            .set_display_name("model-eval-job")
        )
        
        # Import the evaluation result to Vertex AI.
        import_evaluation_task = (
            ModelImportEvaluationOp(
                regression_metrics=evaluation_task.outputs['evaluation_metrics'],
                model=model_resource,
                dataset_type=batch_predictions_input_format,
                dataset_path="", # test_data_gcs_uri
                dataset_paths=ground_truth_gcs_source,
                display_name=eval_display_name,
            )
            .set_display_name("import-model-eval")
        )
```
</details>