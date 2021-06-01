import pandas as pd
import numpy as np
import sklearn
import joblib

from dl_classification_model import config,pipeline
from dl_classification_model import __version__ as _version
from dl_classification_model.processing import utils


def test_load_dataset():
    """ Testing the loading dataset function """

    # Given
    dataset_file_name = config.TESTING_DATA_FILE

    # When
    subject = utils.load_dataset(filename=dataset_file_name)

    # Then
    assert isinstance(subject,pd.DataFrame)
    assert subject.shape == (5940, 41)

def test_save_pipeline():
    """ Testing the save_pipeline function """

    # Given
    # read the data
    data = pd.read_csv(f"{config.DATASET_DIR/config.TRAINING_DATA_FILE}",sep=",", encoding="utf-8")

    # Split in X and y
    X = data.drop(labels=config.TARGET_FEATURE_NAME, axis=1)
    y = data[config.TARGET_FEATURE_NAME]

    # For the 2 classes classification
    y = np.where(y=="functional","functional","non functional or functional needs repair")
    _pipeline = pipeline.pump_pipeline

    _pipeline.fit(X,y)
    subject_file_name_model = f"fake_model_{_version}fake_model"
    subject_file_name_dataprep = f"fake_dataprep{_version}.pkl"
    subject_classes_name = f"{subject_file_name_model}_classes_names_v{_version}.pkl"
    # When
    utils.save_pipeline(_pipeline,save_file_name=subject_file_name_model,dataprep_name = subject_file_name_dataprep)


    # Then
    # Get the files in the model save's directory
    trained_model_dir_file_list = [file.name for file in config.TRAINED_MODEL_DIR.iterdir()]
    
    # Check if the model was saved in TRAINED_MODEL_DIR and with the right filename
    assert subject_file_name_model in trained_model_dir_file_list
    # Check if the dataprep pipeline was saved in TRAINED_MODEL_DIR and with the right filename
    assert subject_file_name_dataprep in trained_model_dir_file_list
    # Check if the model's classes names were saved in TRAINED_MODEL_DIR and with the right filename
    assert subject_classes_name in trained_model_dir_file_list
    # Check if the __init__.py file is in the TRAINED_MODEL_DIR
    assert "__init__.py" in trained_model_dir_file_list
    # Check if the TRAINED_MODEL_DIR folder contains just the new saved pipeline and the __init__.py file
    # remove the fake saves
    utils.rm_tree(config.TRAINED_MODEL_DIR/subject_file_name_model)
 
    (config.TRAINED_MODEL_DIR/subject_file_name_dataprep).unlink()
    (config.TRAINED_MODEL_DIR/subject_classes_name).unlink()


def test_load_pipeline():
    """ Testing the load_pipeline function """

    # Given
    pipeline_file_name = {"dataprep_name": f"{config.DATAPREP_PIPELINE_NAME}{_version}.pkl","model_name" : f"{config.MODEL_PIPELINE_NAME}{_version}"}

    # When
    subject = utils.load_pipeline(file_name= pipeline_file_name)

    # Then
    assert isinstance(subject,sklearn.pipeline.Pipeline)


def test_remove_old_pipelines():
    """ Test the remove_old_pipelines function """

    # Given
    # read the data
    data = pd.read_csv(f"{config.DATASET_DIR/config.TRAINING_DATA_FILE}",sep=",", encoding="utf-8")

    # Split in X and y
    X = data.drop(labels=config.TARGET_FEATURE_NAME, axis=1)
    y = data[config.TARGET_FEATURE_NAME]

    # For the 2 classes classification
    y = np.where(y=="functional","functional","non functional or functional needs repair")
    _pipeline = pipeline.pump_pipeline

    _pipeline.fit(X,y)

    subject_file_name_pipe = f"fake_pipe_path{_version}.pkl"

    # Saving the former pipeline
    save_pipe_path = config.TRAINED_MODEL_DIR/subject_file_name_pipe
    joblib.dump(_pipeline.steps[:-1],save_pipe_path)

    trained_model_dir_file_list = [file.name for file in config.TRAINED_MODEL_DIR.iterdir()]

    assert save_pipe_path.name in trained_model_dir_file_list

    # Saving the subject pipeline
    subject = "subject_pipeline_v_fake.pkl"
    save_subject_test_path = config.TRAINED_MODEL_DIR/subject
    joblib.dump(_pipeline.steps[:-1],save_subject_test_path)

    # When 
    utils.remove_old_pipelines(files_to_keep=[subject])

    trained_model_dir_file_list = [file.name for file in config.TRAINED_MODEL_DIR.iterdir()]

    # Then
    assert subject in trained_model_dir_file_list
    assert save_pipe_path.name not in trained_model_dir_file_list
    assert "__init__.py" in trained_model_dir_file_list 

    # removing the fake pipeline
    save_subject_test_path.unlink()



def test_input_data_is_valid():
    """ Test input_data_is_valid function """

    # read the test data
    test_data = pd.read_csv(config.DATASET_DIR/config.TESTING_DATA_FILE)

    # Given
    # Droping a feature
    test_data_missing_feature = test_data.copy()
    test_data_missing_feature = test_data.drop(labels=config.VARIABLES_TO_KEEP[0], axis=1)

    # Converting a numerical variable to a string
    test_data_num_type = test_data.copy()
    test_data_num_type[config.NUMERICAL_VARIABLES[0]] = str(test_data_num_type[config.NUMERICAL_VARIABLES[0]])

    #When
    # subject with the expected dataset type and features
    subject = utils.input_data_is_valid(input_data=test_data)
    # subject with missing features
    subject_missing = utils.input_data_is_valid(input_data=test_data_missing_feature)
    # subject with the converted numerical variable
    subject_num_type = utils.input_data_is_valid(input_data=test_data_num_type)


    #Then
    assert subject == True
    assert subject_missing == False
    assert subject_num_type == False

