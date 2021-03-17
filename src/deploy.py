import os
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.model import Model

from dotenv import load_dotenv

from configuration.experiment_configuration import ExperimentConfiguration
from helpers.azureml_pipelines import get_workspace_from_env


def deploy():
    ws = get_workspace_from_env()

    model = ws.models[ExperimentConfiguration.model_name]
    transform_pipeline = ws.models[ExperimentConfiguration.transform_pipeline_name]

    entry_script = os.path.join(ExperimentConfiguration.experiment_folder,
                                ExperimentConfiguration.score_script)
    conda_file = os.path.join(
        ExperimentConfiguration.experiment_folder, 'environment.yml')
    inference_config = InferenceConfig(runtime="python",
                                       entry_script=entry_script,
                                       conda_file=conda_file)

    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=ExperimentConfiguration.cpu_cores, memory_gb=ExperimentConfiguration.memory_gb)

    service = Model.deploy(
        ws, ExperimentConfiguration.service_name, [model, transform_pipeline],
        inference_config, deployment_config)

    service.wait_for_deployment(True)
    print(service.state)
    print(service.get_logs())


if __name__ == "__main__":
    load_dotenv()
    deploy()
