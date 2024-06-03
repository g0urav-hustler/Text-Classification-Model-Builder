from src import logger
from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_02_data_processing import DataProcessingPipeline    
from src.pipeline.stage_03_train_model import TrainModelPipeline 
from src.pipeline.stage_04_model_evaluation import EvaluateModelPipeline                                               


class InvokePipeline:
    def __init__(self):
        pass

    def main(self):

        STAGE_NAME = "Data Ingestion Stage"

        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DataIngestionPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e


        STAGE_NAME = "Data Preprocessing Stage"

        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DataProcessingPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e
        
        
        STAGE_NAME = "Model Training Stage"

        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = TrainModelPipeline()
            evaluation_metric = obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e
        
        STAGE_NAME = "Model Evaluation Stage"

        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = EvaluateModelPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e
        
if __name__ == '__main__':
    try:
        obj = InvokePipeline()
        result = obj.main()
    except Exception as e:
        logger.exception(e)
        raise e