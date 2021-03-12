from azureml.core import Experiment, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineData
from azureml.core.runconfig import RunConfiguration

from dotenv import load_dotenv

from configuration.experiment_configuration import ExperimentConfiguration
from helpers.azureml_pipelines import (PipelineBuilder, create_compute,
                                       register_dataset, get_workspace_from_env)

if __name__ == "__main__":
    load_dotenv()
    ws = get_workspace_from_env()
    # Register dataset
    register_dataset(ws, ExperimentConfiguration.dataset_name,
                     [ExperimentConfiguration.dataset_file], ExperimentConfiguration.data_path_name)

    # Create cluster
    pipeline_cluster = create_compute(ws, ExperimentConfiguration.cluster_name,
                                      ExperimentConfiguration.cluster_virtual_machine_size,
                                      ExperimentConfiguration.cluster_virtual_machine_nodes)

    # Create environment
    telecom_env = Environment(ExperimentConfiguration.environment_name)
    packages = CondaDependencies.create(conda_packages=ExperimentConfiguration.conda_packages,
                                        pip_packages=ExperimentConfiguration.pip_packages)
    telecom_env.python.conda_dependencies = packages

    # Create Run Configuration
    pipeline_run_config = RunConfiguration()
    pipeline_run_config.target = pipeline_cluster
    pipeline_run_config.environment = telecom_env

    # Register Pipeline Data
    prepare_data_folder = PipelineData(
        ExperimentConfiguration.prepare_data_output_folder, datastore=ws.get_default_datastore())
    train_folder = PipelineData(
        ExperimentConfiguration.train_output_folder, datastore=ws.get_default_datastore())
    metrics_data = PipelineData(name=ExperimentConfiguration.train_metrics_output,
                                datastore=ws.get_default_datastore())

    # Create pipeline
    pipeline = PipelineBuilder(ws).add_step(
        PythonScriptStep(name="Prepare Data",
                         script_name=ExperimentConfiguration.prepare_data_script,
                         arguments=[
                             '--random-state', 1,
                             '--dataset-name', ExperimentConfiguration.dataset_name,
                             '--prepare-data-folder', prepare_data_folder,
                             '--train-dataset', ExperimentConfiguration.train_dataset_file,
                             '--test-dataset', ExperimentConfiguration.test_dataset_file,
                             '--transform-pipeline-name',
                             ExperimentConfiguration.transform_pipeline_name,
                         ],
                         outputs=[prepare_data_folder],
                         source_directory=ExperimentConfiguration.experiment_folder,
                         runconfig=pipeline_run_config,
                         allow_reuse=True)
    ).add_step(
        PythonScriptStep(name="Train model",
                         script_name=ExperimentConfiguration.training_script,
                         arguments=[
                             '--random-state', 1,
                             '--prepare-data-folder', prepare_data_folder,
                             '--train-folder', train_folder,
                             '--train-dataset', ExperimentConfiguration.train_dataset_file,
                             '--test-dataset', ExperimentConfiguration.test_dataset_file,
                             '--model-name', ExperimentConfiguration.model_name,
                             '--metrics-output', metrics_data,
                         ],
                         inputs=[prepare_data_folder],
                         outputs=[train_folder, metrics_data],
                         source_directory=ExperimentConfiguration.experiment_folder,
                         runconfig=pipeline_run_config,
                         allow_reuse=True)
    ).add_step(
        PythonScriptStep(name="Register model",
                         script_name=ExperimentConfiguration.register_script,
                         arguments=[
                             '--train-folder', train_folder,
                             '--model-name', ExperimentConfiguration.model_name,
                             '--metrics-input', metrics_data,
                         ],
                         inputs=[train_folder, metrics_data],
                         source_directory=ExperimentConfiguration.experiment_folder,
                         runconfig=pipeline_run_config,
                         allow_reuse=True)
    ).build()

    print("Pipeline steps defined")

    # Run Experiment
    experiment = Experiment(
        workspace=ws, name=ExperimentConfiguration.experiment_name)
    pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
    pipeline_run.wait_for_completion(show_output=True)

    # Publish the pipeline from the run
    published_pipeline = pipeline_run.publish_pipeline(
        name=ExperimentConfiguration.pipineline_name,
        description=ExperimentConfiguration.pipineline_description,
        version="1.0")

    print(f"Pipeline endpoint: {published_pipeline.endpoint}")
