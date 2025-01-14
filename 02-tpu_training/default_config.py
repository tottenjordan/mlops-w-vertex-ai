from collections import abc
from ml_collections import config_dict

# from google.cloud.aiplatform import gapic


def get_config() -> config_dict.ConfigDict:
    """Default config for training a ResNet50 on imagenet."""

    config = config_dict.ConfigDict()

    # cloud env
    config.project_id = "hybrid-vertex"
    config.location = "us-central1"
    config.bucket_uri = None
    config.service_account = None
    config.repository = None
    
    # train job
    config.train_tpu = "TPU_V2"
    config.train_tpu_count = 8
    config.train_compute = "cloud-tpu"
    config.tb_instance = None
    config.train_image = "us-central1-docker.pkg.dev/hybrid-vertex/my_tpu_repo/tpu-train:latest"
    config.pipeline_local_json = "UPDATE_ME.json"
    
    # deployment
    config.deploy_gpu = "NVIDIA_TESLA_T4"
    config.deploy_gpu_count = 1
    config.deploy_compute = "n1-standard-4"
    # config.deploy_version = "tf2-gpu.2-13"
    config.deploy_image = "us-docker.pkg.dev/cloud-aiplatform/prediction/tf2-gpu.2-13:latest"
    config.endpoint_id = "4267284891747483648"

    # train args
    config.train_strategy = "tpu"
    config.trainer_args = None
    config.epochs = 10
    config.steps = 10_000
    config.model_display_name = "my-model"
    
    # managed pipeline
    config.pipeline_root = None
    config.pipeline_name = "my-pipeline-name"
    
    # # output dirs
    # config.experiment_name = None
    # config.run_name = None
    # config.experiment_dir = ""
    # config.checkpoint_dir = ""
    # config.base_output_dir = ""
    # config.log_dir = ""
    # config.data_dir = ""
    # config.artifacts_dir = ""

    config.seed = 0
    return config