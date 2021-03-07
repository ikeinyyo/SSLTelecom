from azureml.core.experiment import Experiment
from dotenv import load_dotenv

from helpers.azureml_helpers import (get_workspace_from_env, DummyRun, calculate_k_best, read_and_split_data,
                                     create_transform_pipeline, train_model, add_tags, log_metrics, save_and_register_model)

RUN_LOCAL = False
DATA_FILEPATH = 'data/mobile_phone_info.csv'
OUTPUT_FOLDER = 'output'
MODEL_OUTPUT_FILEPATH = 'mobile_phone.pkl'
MODEL_NAME = 'mobile_price_classificator'
RANDOM_STATE = 1


def main():
    if RUN_LOCAL:
        run = DummyRun()
    else:
        ws = get_workspace_from_env()
        experiment = Experiment(workspace=ws, name='TrainTest')
        run = experiment.start_logging()

    X_train, X_validation, y_train, y_validation, X_test, y_test = read_and_split_data(
        DATA_FILEPATH, RANDOM_STATE)

    k_best = calculate_k_best(
        X_train, y_train, X_validation, y_validation, RANDOM_STATE)
    data_pipeline = create_transform_pipeline(k_best)

    model = train_model(X_train, y_train, X_validation,
                        y_validation, data_pipeline, RANDOM_STATE)

    tags = add_tags(run, model, k_best, data_pipeline)
    metrics = log_metrics(run, model,
                          data_pipeline, X_test, y_test)

    save_and_register_model(run, model, MODEL_NAME,
                            OUTPUT_FOLDER, MODEL_OUTPUT_FILEPATH, tags, metrics)

    run.complete()


if __name__ == "__main__":
    load_dotenv()
    main()
