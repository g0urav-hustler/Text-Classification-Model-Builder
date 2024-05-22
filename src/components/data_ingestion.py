import os
from src.utils.common import copy_files
from src.entity.config_entity import DataIngestionConfig

class DataIngestion():

    def __init__(self, config = DataIngestionConfig):
        self.config = config

    def get_raw_data(self):
        source_dir = self.config.web_data_dir
        target_dir = self.config.raw_data_dir
        files_list = os.listdir(source_dir)
        
        # copy file to raw data
        copy_files(files_list, source_dir, target_dir, file_extension=False)
        
        