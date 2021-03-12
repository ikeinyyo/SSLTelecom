
import json
import numpy as np
import joblib
from azureml.core.model import Model


def init():
    global MODEL
    global TRANSFORM_PIPELINE

    MODEL = joblib.load(Model.get_model_path('mobile_price_classificator'))
    TRANSFORM_PIPELINE = joblib.load(
        Model.get_model_path('transform_pipeline_model'))


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    predictions = MODEL.predict(TRANSFORM_PIPELINE.transform(data))
    return predictions.tolist()
