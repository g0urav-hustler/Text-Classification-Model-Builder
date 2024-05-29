from src.utils.common import load_json, join_path
from src.entity.config_entity import TrainModelConfig
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

class TrainModel:
    def __init__(self, config = TrainModelConfig):
        self.config = config

    def dataset_format(self, data):

        result_data = Dataset.from_dict(data)

        return result_data
    
    
    def compute_metrics(self,pred):
        
        labels = pred.label_ids

        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

        acc = accuracy_score(labels, preds)

        # Return the computed metrics as a dictionary
        return {
            'Accuracy': acc,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        }


    def train_model(self):
        config = self.config

        train_data = load_json(Path(join_path(config.train_data_path, "train_data.json")))
        train_data = self.dataset_format(train_data)

        val_data = load_json(Path(join_path(config.val_data_path, "val_data.json")))
        val_data = self.dataset_format(val_data)


        training_args = TrainingArguments(
            output_dir = join_path(self.config.save_model_dir, self.config.model_name),
            num_train_epochs = self.config.epochs,
            per_device_train_batch_size= self.config.train_batch_size,
            per_device_eval_batch_size=self.config.val_batch_size,
            learning_rate = 2e-5,
            disable_tqdm = False
        )


        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels = config.num_labels)
        
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_data,
            eval_dataset = val_data,
            compute_metrics = self.compute_metrics
        )

        # model training
        trainer.train()

        trainer.save_model(self.config.save_model_path)
        

