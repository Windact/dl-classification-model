import pandas as pd
import logging

from dl_classification_model import __version__ as _version
from dl_classification_model import config
from dl_classification_model.processing import utils

_logger = logging.getLogger(__name__)


def make_prediction(input_data):
    """ Predict the class of the water pumps 
    
    Parameters
    ----------
    input_data : dict
         A dictionary containing features (keys) expected for by the pipeline and its corresponding values
    
    Returns
    -------
    dict
    Returns a dictionary containing the predictions and the version of the pipeline
    """

    # Converting the input_data dict to a pd.DataFrame
    data = pd.DataFrame(input_data)
    # Checking if the data is valid
    if utils.input_data_is_valid(data):
        # loading the latest fitted pipeline
        pipe_line_file_name = f"{config.MODEL_PIPELINE_NAME}{_version}.pkl"
        _pipe_pump = utils.load_pipeline(file_name=pipe_line_file_name)
        # Predictions
        outputs = _pipe_pump.predict(data)

        results = {"predictions" : outputs, "version" : _version}

        _logger.info(
            f"Making predictions with the model version : {_version}"
            f"Inputs : {data}"
            f"predictions : {results}"
        )
        return results