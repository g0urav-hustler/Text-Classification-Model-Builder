import torch
from src.utils.common import load_json
from src.entity.config_entity import EvaluateModelConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification

class EvaluateModel:
    def __init__(self, config = EvaluateModelConfig):
        self.config = config

    def get_predictions(self,input_ids):
        predicted_labels = []
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(self.config.pretrained_model_path)
        for i in range(len(input_ids)):
            predicted_input = torch.tensor([input_ids[i]])
            preds = pretrained_model(predicted_input)
            predicted_labels.append(preds.logits.argmax())
            
        return predicted_labels
    
    def get_evaluation_report(self, actual_labels, predicted_labels):
        precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='macro')

        acc = accuracy_score(actual_labels, predicted_labels)
        report = {
            'Accuracy': acc,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        }

        return report

    def get_model_evaluation(self):

        test_data = load_json(self.config.test_data_path)
        input_ids = test_data["input_ids"]

        predicted_labels = self.get_predictions(input_ids)
        actual_labels = test_data["labels"]

        model_evaluation_result = self.get_evaluation_report(actual_labels, predicted_labels)

        return model_evaluation_result
        