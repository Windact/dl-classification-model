import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import KBinsDiscretizer,MinMaxScaler
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.encoding import OneHotEncoder
from feature_engine.discretisation import EqualWidthDiscretiser
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import logging

# Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

# Local source tree imports
from  dl_classification_model.processing import  preprocessors as pp
from dl_classification_model import config

_logger = logging.getLogger(__name__)

def create_model(beta_1 = 0.8,beta_2=0.999,learning_rate = 0.01,drop_rate_input= 0.2, drop_rate_hidden = 0.3,weight_constraint = 3,units = 80, seed = config.SEED):
    model = Sequential()
    model.add(Dropout(drop_rate_input,seed= seed))
    model.add(Dense(units=units,activation='relu',kernel_constraint=MaxNorm(max_value=weight_constraint, axis=0)))
    model.add(Dropout(drop_rate_hidden,seed= seed))
    model.add(Dense(units=units/2,activation='relu',kernel_constraint=MaxNorm(max_value=weight_constraint, axis=0)))
    model.add(Dropout(drop_rate_hidden,seed= seed))
    model.add(Dense(units=1,activation='sigmoid'))

    # For a binary classification problem
    optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=1e-07, amsgrad=False,name='Adam')
    model.compile(loss='binary_crossentropy',metrics=["accuracy"], optimizer=optimizer)
    
    return model

# callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,mode = "auto",verbose=1,patience=10, min_lr=0.01)
early_stop = EarlyStopping(monitor='val_loss', mode='min',min_delta=0, verbose=1, patience=20)


pump_pipeline =Pipeline(steps=[("feature_to_keeper",pp.FeatureKeeper(variables_to_keep=config.VARIABLES_TO_KEEP)),
                         ("missing_imputer", pp.MissingImputer(numerical_variables=config.NUMERICAL_VARIABLES)),
                         ("yeoJohnson",YeoJohnsonTransformer(variables=config.YEO_JHONSON_VARIABLES)),
                         ("discretization",EqualWidthDiscretiser(bins=5, variables=config.NUMERICAL_VARIABLES)),
                         ("categorical_grouper",pp.CategoricalGrouping(config_dict=config.VARIABLES_TO_GROUP)),
                         ("rareCategories_grouper",pp.RareCategoriesGrouping(threshold=config.VARIABLES_THRESHOLD)),
                         ("one_hot_encoder",OneHotEncoder(variables=config.REAL_CATEGORICAL_VARIABLES,drop_last=False)),
                         ("scaler",MinMaxScaler()),
                         ("model",KerasClassifier(build_fn=create_model,epochs=1,validation_split=0.2,batch_size= 256, verbose=1,callbacks=[early_stop,reduce_lr],shuffle = True))])