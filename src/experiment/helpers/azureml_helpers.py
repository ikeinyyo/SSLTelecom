import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, f1_score, plot_confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def create_transform_pipeline(k_best):
    return Pipeline([
        ('k_best_selector', SelectKBest(f_classif, k=k_best))
    ])


def calculate_k_best(X_train, y_train, random_state):
    svm = SVC(random_state=random_state)
    accuracy_list_train = []
    k_best = np.arange(1, 21, 1)

    for each in k_best:
        selector = SelectKBest(f_classif, k=each)
        X_train_prepared = selector.fit_transform(X_train, y_train)
        svm.fit(X_train_prepared, y_train)
        accuracy_list_train.append(svm.score(X_train_prepared, y_train))
    data = pd.DataFrame(
        data={'best features number': k_best, 'train_score': accuracy_list_train})
    return data["train_score"].idxmax() + 1


def train_model(X_train, y_train, random_state):
    C = [1]
    kernel = ["linear"]
    gamma = ["auto"]
    decision_function_shape = ["ovo"]

    svm = SVC(random_state=random_state)
    grid_svm = GridSearchCV(estimator=svm, cv=3, param_grid=dict(
        kernel=kernel, C=C, gamma=gamma, decision_function_shape=decision_function_shape))
    grid_svm = grid_svm.fit(X_train, y_train)
    return grid_svm.best_estimator_


def register_model(run, source_folder, model_name, metrics=None, tags=None):
    model_pickle = f'{model_name}.pkl'
    model_path = os.path.join(source_folder, model_pickle)
    run.upload_file(model_pickle, model_path)
    run.register_model(model_path=model_pickle,
                       model_name=model_name,
                       properties=metrics, tags=tags)


def save_and_register_model(run, model, output_folder, model_name, metrics=None, tags=None):
    save_pickle_data(output_folder, model_name, model)
    register_model(run, output_folder, model_name, metrics, tags)


def add_tags(run, svm):
    tags = {
        'C': svm.C,
        'decision_function_shape': svm.decision_function_shape,
        'gamma': svm.gamma,
        'kernel': svm.kernel,
    }

    for (key, value) in tags.items():
        run.tag(key, value)

    return tags


def calculate_metrics(svm, X_test, y_test):
    y_predicted = svm.predict(X_test)
    return {
        'recall': recall_score(
            y_test, y_predicted, average='macro'),
        'precision': precision_score(
            y_test, y_predicted, average='macro'),
        'accuracy': accuracy_score(
            y_test, y_predicted),
        'f1': f1_score(
            y_test, y_predicted, average='macro'),
    }


def log_confusion_matrix(run, svm, X_test, y_test):
    plot_confusion_matrix(svm, X_test, y_test)
    run.log_image(name='confusion_matrix', plot=plt)


def log_metrics(run, svm, X_test, y_test):
    metrics = calculate_metrics(svm, X_test, y_test)

    for (key, value) in metrics.items():
        run.log(key, value)

    log_confusion_matrix(run, svm, X_test, y_test)

    return metrics


def save_pickle_data(folder, key, data):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{key}.pkl")
    joblib.dump(value=data, filename=filename)
    return filename


def load_pickle_data(folder, key):
    return joblib.load(filename=os.path.join(folder, f"{key}.pkl"))
