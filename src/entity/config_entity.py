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