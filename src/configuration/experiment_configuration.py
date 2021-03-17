class ExperimentConfiguration:
    # Data configuration
    dataset_name = "mobile_phones_info"
    dataset_file = "data/mobile_phone_info.csv"
    data_path_name = "mobile_phones_data"
    model_name = "mobile_price_classificator"
    transform_pipeline_name = "transform_pipeline_model"

    # Cluster configuration
    cluster_name = "training-cluster"
    cluster_virtual_machine_size = "Standard_DS1_v2"
    cluster_virtual_machine_nodes = 1

    # Pipeline configuration
    pipineline_name = "train-pipeline"
    pipineline_description = "A pipeline to train and register model"
    experiment_name = "Train"
    environment_name = 'telecom'
    experiment_folder = 'experiment'
    conda_packages = ['pip=19.2.2']
    pip_packages = ['pandas==1.0.3', 'matplotlib==3.2.1', 'sklearn==0.0',
                    'azureml==0.2.7', 'azureml-core==1.24.0', 'azureml-dataset-runtime=1.24.0']

    # Scripts configuration
    prepare_data_output_folder = 'output_prepare_data'
    train_output_folder = 'output_train'
    train_metrics_output = 'metrics_output'
    prepare_data_script = 'mobile_prepare.py'
    training_script = 'mobile_training.py'
    register_script = 'mobile_register.py'
    train_dataset_file = 'train.csv'
    test_dataset_file = 'test.csv'

    # Deploy
    score_script = 'score.py'
    service_name = 'mobile-service'
    cpu_cores = 1
    memory_gb = 1
