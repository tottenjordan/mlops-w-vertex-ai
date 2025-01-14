from google_cloud_pipeline_components.types import artifact_types
from kfp.v2.dsl import component
from typing import NamedTuple
@component(
  base_image='python:3.10.15',
  packages_to_install=['google-cloud-aiplatform==1.71.0'],
)
def create_next_model_version(
    parent_model: str,
    artifact_uri: str,
    serving_container: str,
    project: str,
    region: str,
    pipeline_version: str,
    version_alias: str,
) -> NamedTuple('Outputs', [
    ('model', artifact_types.VertexModel),
    ('model_resource_name', str),
]):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    model = aiplatform.Model.upload(
        display_name=f"taxi-{version}",
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container,
        parent_model=parent_model,
        is_default_version=True,
        version_aliases=[version_alias],
        # version_description="This is the second version of the model",
    )

    return (
        model,
        model.resource_name,
    )