import argparse
from azureml.core import Run

from helpers.azureml_helpers import load_pickle_data, register_model


def run_register(args):
    run = Run.get_context()

    metrics = load_pickle_data(args.metrics_input, 'metrics')
    register_model(run, args.train_folder, args.model_name, metrics)

    run.complete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-folder", type=str,
                        dest='train_folder', help='training data folder')
    parser.add_argument("--model-name", type=str,
                        dest='model_name', help='training dataset')
    parser.add_argument("--metrics-input", type=str,
                        dest='metrics_input', help='Input for metrics')
    run_register(parser.parse_args())
