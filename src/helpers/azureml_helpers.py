from azureml.core import Workspace
from azureml.core.experiment import Experiment
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os


def get_workspace_from_env():
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace_name = os.getenv("WORKSPACE_NAME")

    try:
        return Workspace.get(subscription_id=subscription_id,
                             resource_group=resource_group, name=workspace_name)
    except Exception as e:
        print("Workspace not accessible. Change your parameters or create a new workspace below")
        raise e


def read_and_split_data(dataset_filepath, random_state):
    df = pd.read_csv(dataset_filepath)

    y = df['price_range']
    X = df.drop(['price_range'], axis=1)

    X, X_test, y, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.30, random_state=random_state)

    return X_train, X_validation, y_train, y_validation, X_test, y_test


def create_transform_pipeline(k_best):
    return Pipeline([
        ('k_best_selector', SelectKBest(f_classif, k=k_best))
    ])


def calculate_k_best(X_train, y_train, X_validation, y_validation, random_state):
    svm = SVC(random_state=random_state)
    accuracy_list_train = []
    k = np.arange(1, 21, 1)

    for each in k:
        selector = SelectKBest(f_classif, k=each)
        X_train_prepared = selector.fit_transform(X_train, y_train)
        X_validation_prepared = selector.fit_transform(
            X_validation, y_validation)
        svm.fit(X_train_prepared, y_train)
        accuracy_list_train.append(
            svm.score(X_validation_prepared, y_validation))
    data = pd.DataFrame(
        data={'best features number': k, 'train_score': accuracy_list_train})
    return data["train_score"].idxmax() + 1


def train_model(X_train, y_train, X_validation, y_validation, data_pipeline, random_state):
    X_prepared = data_pipeline.fit_transform(X_train, y_train)

    C = [1]
    kernel = ["linear"]
    gamma = ["auto"]
    decision_function_shape = ["ovo"]

    svm = SVC(random_state=random_state)
    grid_svm = GridSearchCV(estimator=svm, cv=3, param_grid=dict(
        kernel=kernel, C=C, gamma=gamma, decision_function_shape=decision_function_shape))
    grid_svm = grid_svm.fit(X_prepared, y_train)
    return grid_svm.best_estimator_


def add_tags(run, svm, k_best, data_pipeline):
    tags = {
        'transforms': f"{data_pipeline.steps}",
        'k_best': k_best,
        'C': svm.C,
        'decision_function_shape': svm.decision_function_shape,
        'gamma': svm.gamma,
        'kernel': svm.kernel,
    }
    [run.tag(key, value) for (key, value) in tags.items()]
    return tags


def save_and_register_model(run, model, model_name, output_folder, model_output_filename, tags, metrics):
    os.makedirs(output_folder, exist_ok=True)
    model_path = os.path.join(
        output_folder, model_output_filename)
    joblib.dump(value=model, filename=model_path)
    run.upload_file(model_path, model_path)
    run.register_model(model_path=model_path, model_name=model_name,
                       tags=tags,
                       properties=metrics)


def log_metrics(run, svm, data_pipeline, X, y):
    X = data_pipeline.transform(X)
    y_predicted = svm.predict(X)
    metrics = {
        'recall': recall_score(
            y, y_predicted, average='macro'),
        'precision': precision_score(
            y, y_predicted, average='macro'),
        'accuracy': accuracy_score(
            y, y_predicted),
        'f1': f1_score(
            y, y_predicted, average='macro'),
    }
    [run.log(key, value) for (key, value) in metrics.items()]

    plot_confusion_matrix(svm, X, y)
    run.log_image(name='confusion_matrix', plot=plt)

    return metrics


class DummyRun:
    def start_logging(self):
        print("[START LOGGING]")

    def complete(self):
        print("[COMPLETE]")

    def tag(self, name, value):
        print(f"[TAG] {name}: {value}")

    def log_image(self, name, plot):
        print("[LOG IMAGE] {name}: {plot}")

    def log(self, name, value):
        print(f"[LOG] {name}: {value}")

    def upload_file(self, name, path_or_stream):
        print(f"[UPLOAD FILE] {name}: {path_or_stream}")

    def register_model(self, model_path, model_name, tags, properties):
        print(
            f"[REGISTER MODEL] {model_name}: {model_path}\n[TAGS]: {tags}\n[PROPERTIES]: {properties}")
