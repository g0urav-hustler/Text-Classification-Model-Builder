from dataclasses import dataclass
from pathlib import Path

# creating a dataigesionconfig class
@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_dir : Path
    web_data_dir : Path

# creatign a data preprocessingconfig class
@dataclass(frozen= True)
class DataProcessingConfig:
    model_name : str
    raw_data_dir: Path
    processed_data_dir: Path
    split_data_dir: Path

    text_col: str
    label_col: str

    train_data_size: float
    test_data_size: float
    val_data_size: float
    saved_tokenizer_path: Path

@dataclass(frozen= True)
class TrainModelConfig:
    train_data_path: Path 
    val_data_path: Path
    save_model_dir: Path
    output_dir: Path
    
    model_name : str
    num_labels : int
    epochs : int
    train_batch_size : int
    val_batch_size : int

@dataclass(frozen= True)
class EvaluateModelConfig:
    test_data_path: Path 
    pretrained_model_path : Path