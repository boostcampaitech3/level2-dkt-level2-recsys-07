import os

import torch
import wandb
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import setSeeds

from sklearn.model_selection import KFold

def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    if args.split_method == "user":
        train_data, valid_data = preprocess.split_data(train_data)
        wandb.init(project="dkt", config=vars(args))
        trainer.run(args, train_data, valid_data, list())
        
    elif args.split_method == "k-fold":
        n_splits = args.n_splits
        kfold_auc_list = list()
        kf = KFold(n_splits=n_splits)

        ## -- KFold Training
        for k_th, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
            train_set = torch.utils.data.Subset(train_data, indices = train_idx) # KFold에서 나온 인덱스로 훈련 셋 생성
            val_set = torch.utils.data.Subset(train_data, indices = valid_idx) # KFold에서 나온 인덱스로 검증 셋 생성

            wandb.init(project="dkt", config=vars(args))
            trainer.run(args, train_set, val_set, kfold_auc_list)
            
        ##--------------------KFold 결과 출력----------------------
        for i in range(n_splits):
            print(f"Best AUC for {i+1}th fold is : {kfold_auc_list[i]}")
        print(f"The Average AUC of the model is : {sum(kfold_auc_list) / n_splits:.4f}")



if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
