import os
import argparse
import pandas as pd
from azureml.core import Run

from helpers.azureml_helpers import train_model, add_tags, log_metrics, save_pickle_data


def run_train(args):
    run = Run.get_context()

    # Reading data
    df = pd.read_csv(os.path.join(
        args.prepare_data_folder, args.train_dataset))
    y_train = df['price_range']
    X_train = df.drop(['price_range'], axis=1)

    df = pd.read_csv(os.path.join(args.prepare_data_folder, args.test_dataset))
    y_test = df['price_range']
    X_test = df.drop(['price_range'], axis=1)

    # Training and log metrics
    model = train_model(X_train, y_train, args.random_state)
    add_tags(run, model)
    metrics = log_metrics(run, model, X_test, y_test)

    # Saving data
    save_pickle_data(args.metrics_output, 'metrics', metrics)
    save_pickle_data(args.train_folder, args.model_name, model)

    run.complete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int,
                        dest='random_state', help='training dataset')
    parser.add_argument('--prepare-data-folder', type=str, dest='prepare_data_folder',
                        default='prepare data folder input', help='Folder for results')
    parser.add_argument("--train-folder", type=str,
                        dest='train_folder', help='training data folder')
    parser.add_argument("--train-dataset", type=str,
                        dest='train_dataset', help='train dataset filepath destination')
    parser.add_argument("--test-dataset", type=str,
                        dest='test_dataset', help='test dataset filepath destination')
    parser.add_argument("--model-name", type=str,
                        dest='model_name', help='training dataset')
    parser.add_argument("--metrics-output", type=str,
                        dest='metrics_output', help='Output for metrics')

    run_train(parser.parse_args())
