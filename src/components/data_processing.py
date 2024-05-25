import os
import pandas as pd
from pathlib import Path
from src.utils.common import join_path, save_json, load_json
from src.entity.config_entity import DataProcessingConfig
from transformers import AutoTokenizer

class DataProcessing:
    def __init__(self,config = DataProcessingConfig) -> None:
        self.config = config


    def get_encoded_labels(self, data):
        
        result_data = data.astype('category').cat.codes
        return result_data


    def get_encoded_text(self, data_list):

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        encoded_data = tokenizer(data_list, padding = True, truncation= True )
        
        return encoded_data


    def get_processed_data(self):
        file_name = os.listdir(self.config.raw_data_dir)[0]
        file_path = join_path(self.config.raw_data_dir,file_name)

        df = pd.read_csv(file_path, nrows= 100)
        
        # encoding text

        text_list = df[self.config.text_col].to_list()
        encoded_text = self.get_encoded_text(text_list)

        # encoding the labels
        encoded_labels = self.get_encoded_labels(df[self.config.label_col])
        encoded_labels = encoded_labels.to_list()

        # adding labels
        processed_data_dict = { 
            "input_ids": encoded_text["input_ids"],
            "token_type_ids": encoded_text["token_type_ids"],
            "attention_mask": encoded_text["attention_mask"],
            "labels": encoded_labels
                }
        
        
        save_json(Path(join_path(self.config.processed_data_dir,"processed_data.json")), processed_data_dict)

    def get_range_data(self,data, start_range, end_range):
        for k in data.keys():
            data[k] = data[k][start_range: end_range]

        return data
    
    def get_split_data(self):

        processed_data = load_json(Path(join_path(self.config.processed_data_dir, "processed_data.json")))


        data_len = len(processed_data["labels"])
        train_start_index, train_end_index = 0, int(self.config.train_data_size*data_len)
        val_start_index, val_end_index = train_end_index , train_end_index + int(self.config.val_data_size*data_len)
        test_start_index, test_end_index = val_end_index , data_len

        train_data = self.get_range_data(processed_data.copy(), train_start_index, train_end_index)
        val_data = self.get_range_data(processed_data.copy(),val_start_index, val_end_index)
        test_data = self.get_range_data(processed_data.copy(), test_start_index, test_end_index)

        # save train_data
        save_json(Path(join_path(self.config.split_data_dir, "train_data.json")), train_data)

        #save val data
        save_json(Path(join_path(self.config.split_data_dir, "val_data.json")), val_data)

        #save test data
        save_json(Path(join_path(self.config.split_data_dir, "test_data.json")), test_data)


