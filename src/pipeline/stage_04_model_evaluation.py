from src.config.configuration import ConfigurationManager
from src.components.evaluate_model import EvaluateModel
from src import logger

STAGE_NAME = "Model Evaluation Stage"

class EvaluateModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluate_model_config = config.get_evaluate_model_config()

        evaluate_model = EvaluateModel(evaluate_model_config)
        evaluation_report = evaluate_model.get_model_evaluation()

        print("Evaluation Report == ", evaluation_report)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluateModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e