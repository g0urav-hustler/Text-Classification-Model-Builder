from src.config.configuration import ConfigurationManager
from src.components.train_model import TrainModel
from src import logger

STAGE_NAME = "Model Training Stage"

class DataProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        train_model_config = config.get_train_model_config()


        train_model = TrainModel(train_model_config)
        train_model.train_model()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataProcessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e