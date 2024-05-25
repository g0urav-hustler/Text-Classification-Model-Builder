import os
from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig,
                                      DataProcessingConfig)


class ConfigurationManager:

    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.raw_data_dir])

        data_ingestion_config = DataIngestionConfig(
            raw_data_dir= config.raw_data_dir,
            web_data_dir= config.web_data_dir
        )

        return data_ingestion_config
    
    def get_data_processing_config(self) -> DataProcessingConfig:

        config = self.config.data_processing
        params = self.params.data_processing
        model_name = self.params.train_model.model_name

        create_directories([config.processed_data_dir, config.split_data_dir])

        data_processing_config = DataProcessingConfig(

            model_name = model_name,
            raw_data_dir= config.raw_data_dir,
            processed_data_dir= config.processed_data_dir,
            split_data_dir= config.split_data_dir,

            text_col= params.text_col,
            label_col= params.label_col,

            train_data_size= params.train_data_size,
            test_data_size= params.test_data_size,
            val_data_size= params.val_data_size,

        )

        return data_processing_config