from dataclasses import dataclass
from pathlib import Path

# creating a dataigesionconfig class
@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_dir : Path
    web_data_dir : Path