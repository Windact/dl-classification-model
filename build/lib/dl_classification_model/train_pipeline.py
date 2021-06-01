import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import random as python_random
import logging
import joblib

from dl_classification_model.config import config
from dl_classification_model import pipeline
from dl_classification_model import __version__ as _version 
from dl_classification_model.processing import utils

_logger = logging.getLogger(__name__)

# Making training reproductable

# Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(config.SEED)
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(config.SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(config.SEED)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(config.SEED)

def run_training():
    """ Train the model """

    # read the data
    data = utils.load_dataset(config.TRAINING_DATA_FILE)

    # Split in X and y
    X = data.drop(labels=config.TARGET_FEATURE_NAME, axis=1)
    y = data[config.TARGET_FEATURE_NAME]

    # For the 2 classes classification
    y = np.where(y=="functional","functional","non functional or functional needs repair")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=config.SEED,test_size=0.2)


    # Training wtih gridsearch
    params = {"model__batch_size":[256,500],
          "model__beta_1":[0.8,0.95],
          "model__learning_rate": [0.01,0.001],
          "model__drop_rate_input":[0,0.3],
          "model__drop_rate_hidden":[0.2,0.3],
          "model__units":[80,100]}
    clf = GridSearchCV(estimator=pipeline.pump_pipeline, param_grid=params, scoring='neg_log_loss',n_jobs=-1,cv=3,refit=True,verbose=4)
    clf.fit(X_train, y_train)
    best_params_file = f"best_params_v{_version}.pkl"
    joblib.dump(clf.best_params_,f"{config.TRAINED_MODEL_DIR / best_params_file}")
    _logger.info(f"Best parameters : {clf.best_params_}")
    
    #utils.show_results(clf,X_train_transformed,X_test_transformed,y_train,y_test)

    # Report
    utils.show_results(clf.best_estimator_,X_train,X_test,y_train,y_test)
    
    # Saving the model pipeline
    utils.save_pipeline(clf.best_estimator_)

if __name__ == '__main__':
    run_training()


