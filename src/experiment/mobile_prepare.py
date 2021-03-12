import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from azureml.core import Run, Dataset

from helpers.azureml_helpers import (calculate_k_best,
                                     create_transform_pipeline, save_and_register_model)


def run_prepare_data(args):
    run = Run.get_context()
    ws = run.experiment.workspace

    # Loading data
    df = Dataset.get_by_name(
        ws, name=args.training_dataset_name).to_pandas_dataframe()

    y = df['price_range']
    X = df.drop(['price_range'], axis=1)

    # Creating and registering transform pipeline
    k_best = calculate_k_best(X, y, args.random_state)
    data_pipeline = create_transform_pipeline(k_best)
    run.tag('k_best', k_best)
    run.tag('transforms', f"{data_pipeline.steps}")

    # Preparing data
    prepared_df = pd.DataFrame(data_pipeline.fit_transform(df[X.columns], y))
    X_train, X_test, y_train, y_test = train_test_split(
        prepared_df, y, test_size=0.25, random_state=args.random_state)

    X_train_prepared = pd.DataFrame(X_train)
    X_train_prepared['price_range'] = y_train
    X_test_prepared = pd.DataFrame(X_test)
    X_test_prepared['price_range'] = y_test

    # Saving data
    save_folder = args.prepare_data_folder
    os.makedirs(save_folder, exist_ok=True)
    X_train_prepared.to_csv(os.path.join(
        save_folder, args.train_dataset), index=False, header=True)
    X_test_prepared.to_csv(os.path.join(
        save_folder, args.test_dataset), index=False, header=True)

    # Saving transform pipeline
    save_and_register_model(run, data_pipeline, args.prepare_data_folder,
                            args.transform_pipeline_name, {'k_best': k_best})

    run.complete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int,
                        dest='random_state', help='Random state')
    parser.add_argument("--dataset-name", type=str,
                        dest='training_dataset_name', help='Raw dataset name')
    parser.add_argument('--prepare-data-folder', type=str, dest='prepare_data_folder',
                        default='prepare data folder output', help='Folder for results')
    parser.add_argument("--train-dataset", type=str,
                        dest='train_dataset', help='Train dataset filepath destination')
    parser.add_argument("--test-dataset", type=str,
                        dest='test_dataset', help='Test dataset filepath destination')
    parser.add_argument("--transform-pipeline-name", type=str,
                        dest='transform_pipeline_name', help='Transform pipeline name')
    run_prepare_data(parser.parse_args())
