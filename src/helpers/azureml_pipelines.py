import os
from azureml.core import Dataset
from azureml.pipeline.core import Pipeline
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Workspace


class PipelineBuilder():
    def __init__(self, ws):
        self.ws = ws
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)
        return self

    def build(self):
        return Pipeline(workspace=self.ws, steps=self.steps)


def register_dataset(ws, dataset_name, files, target_path):
    if dataset_name in ws.datasets:
        print('Dataset already registered.')
        return

    default_ds = ws.get_default_datastore()
    default_ds.upload_files(
        files=files, target_path=target_path, overwrite=True, show_progress=True)

    tab_data_set = Dataset.Tabular.from_delimited_files(
        path=(default_ds, f'{target_path}/*.csv'))

    tab_data_set = tab_data_set.register(workspace=ws,
                                         name=dataset_name,
                                         description=f'{dataset_name} data',
                                         tags={'format': 'CSV'},
                                         create_new_version=True)
    print('Dataset registered.')


def create_compute(ws, cluster_name, vm_size, max_nodes):
    try:
        return ComputeTarget(workspace=ws, name=cluster_name)
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, max_nodes=max_nodes)
        training_cluster = ComputeTarget.create(
            ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
        return training_cluster


def get_workspace_from_env():
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace_name = os.getenv("WORKSPACE_NAME")

    try:
        return Workspace.get(subscription_id=subscription_id,
                             resource_group=resource_group, name=workspace_name)
    except Exception as exception:
        print("Workspace not accessible. Change your parameters or create a new workspace below")
        raise exception
