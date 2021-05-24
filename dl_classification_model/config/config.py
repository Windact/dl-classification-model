import pandas as pd
import pathlib

import dl_classification_model

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(dl_classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TRAINING_DATA_FILE = "train.csv"
TESTING_DATA_FILE = "test.csv"
TARGET_FEATURE_NAME = "status_group"

MODEL_PIPELINE_NAME = "neural_net_pipe_v"


# Seed for random state
SEED = 42
# Model params
BEST_PARAMS = {'batch_size': 128,
 'beta_1': 0.8,
 'beta_2': 0.999,
 'drop_rate_hidden': 0.3,
 'drop_rate_input': 0.2,
 'learning_rate': 0.01,
 'units': 80,
 'weight_constraint': 3}

# Model accpetance threshold
ACCEPTABLE_MODEL_DIFFERENCE = 0.05

VARIABLES_THRESHOLD = {
    'scheme_management': 0.04,
    'extraction_type_class': 0.06,
    'management_group':0.06,
    'quality_group':0.08,
    'source_type':0.17,
    'waterpoint_type_group':0.1,
    'quantity_group':0.1
}

VARIABLES_TO_KEEP = ['amount_tsh', 'gps_height','construction_year',
       'population', 'region', 'basin',
       'public_meeting', 'scheme_management', 'permit',
       'extraction_type_class', 'management_group', 'payment_type',
       'quality_group', 'quantity_group', 'source_type',
       'waterpoint_type_group']

NUMERICAL_VARIABLES = ['amount_tsh', 'gps_height','construction_year','population']
YEO_JHONSON_VARIABLES = ['amount_tsh', 'gps_height','population']

VARIABLES_TO_GROUP = {
    'region':{
        'Dodoma,Singida':["Dodoma","Singida"],
        'Mara,Tabora,Rukwa,Mtwara,Lindi': ["Mara","Tabora","Rukwa","Mtwara","Lindi"],
        'Manyara,Dar es Salaam,Tanga' : ["Manyara","Dar es Salaam","Tanga"]
    }
}


# Categorical variables (no boolean)
REAL_CATEGORICAL_VARIABLES = ['region',
 'basin',
 'scheme_management',
 'extraction_type_class',
 'management_group',
 'payment_type',
 'quality_group',
 'quantity_group',
 'source_type',
 'waterpoint_type_group']

