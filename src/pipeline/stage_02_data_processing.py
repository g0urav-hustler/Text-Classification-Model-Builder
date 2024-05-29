from src.config.configuration import ConfigurationManager
from src.components.data_processing import DataProcessing
from src import logger

STAGE_NAME = "Data Processing Stage"

class DataProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config =ConfigurationManager()
        data_processing_config = config.get_data_processing_config()

        data_processing = DataProcessing(data_processing_config)

        data_processing.get_processed_data()
        data_processing.get_split_data()
        data_processing.save_tokenizer()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataProcessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e