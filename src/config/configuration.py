import os
from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig,
                                      DataProcessingConfig,
                                      TrainModelConfig,
                                      EvaluateModelConfig)


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
        model_name = self.params.model_params.model_name
        saved_tokenizer_path = Path(os.path.join(config.saved_tokenizer_dir, model_name, "tokenizer"))


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
            saved_tokenizer_path= saved_tokenizer_path 

        )

        return data_processing_config
    
    def get_train_model_config(self) -> TrainModelConfig:

        num_labels = self.params.num_labels
        config = self.config.train_model
        params = self.params.model_params
        model_path = Path(os.path.join(config.saved_model_dir, params.model_name ,"model"))
        
        
        create_directories([config.saved_model_dir])

        train_model_config = TrainModelConfig(
            train_data_path= config.train_data_path,
            val_data_path= config.val_data_path,
            save_model_path= model_path,
            output_dir = config.output_dir,

            model_name = params.model_name,
            num_labels = num_labels,
            epochs = params.epochs,
            train_batch_size = params.train_batch_size,
            val_batch_size = params.val_batch_size
        )

        return train_model_config
    
    def get_evaluate_model_config(self) -> EvaluateModelConfig:

        config = self.config.evaluate_model
        params = self.params.model_params
        model_path = Path(os.path.join(config.saved_model_dir, params.model_name ,"model"))
        
        evaluate_model_config = EvaluateModelConfig(
            test_data_path =Path(config.test_data_path),
            pretrained_model_path = model_path
        )

        return evaluate_model_config