from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from logging import getLogger
from recbole.model.general_recommender import LightGCN, EASE, MultiVAE
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model, get_trainer

if __name__ == '__main__':
    config = Config(model='LightGCN', config_file_list=['config.yaml'])
    #init logger
    init_logger(config)
    logger = getLogger()
    
    # write config info into log
    logger.info(config)

    dataset = create_dataset(config)
    
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = LightGCN(config, train_data.dataset).to(config['device'])    

    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
