import os

import torch
import wandb
import numpy as np
import random

from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess, get_loaders
from dkt.utils import setSeeds

from dkt.pseudoLabelTrainer import Trainer, PseudoLabel

from sklearn.model_selection import KFold

def main(args):
    #wandb.login()
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)

    train_data = preprocess.get_train_data()
    
    # pseudo labeling
    if(args.pseudo == True):
        pseudo = PseudoLabel(Trainer())
        
        # split train, valid, test
        train_data, valid_data, test_data = preprocess.pseudo_split_data(train_data)
        
        # pseudo labeling for n time
        pseudo.run(args.n_pseudo, args, train_data, valid_data, test_data)

        train_data = pseudo.get_pseudo_train_data() # pseudo labeled test + train


    # wandb.init(project="dkt", config=vars(args))
        
    if args.split_method == "user":
        train_data, valid_data = preprocess.split_data(train_data)
        trainer.run(args, train_data, valid_data, list())

    elif args.split_method == "k-fold":
        n_splits = args.n_splits
        kfold_auc_list = list()
        kf = KFold(n_splits=n_splits)

        ## -- Making train_data from pseudo label
        if args.pseudo == True:
            train_data = np.concatenate((train_data, valid_data))
            random.shuffle(train_data) # concat and shuffle

        ## -- KFold Training
        for k_th, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
            train_set = torch.utils.data.Subset(train_data, indices = train_idx) # KFold에서 나온 인덱스로 훈련 셋 생성
            val_set = torch.utils.data.Subset(train_data, indices = valid_idx) # KFold에서 나온 인덱스로 검증 셋 생성

            trainer.run(args, train_set, val_set, kfold_auc_list,k_th+1)
            
        ##--------------------KFold 결과 출력----------------------
        for i in range(n_splits):
            print(f"Best AUC for {i+1}th fold is : {kfold_auc_list[i]}")
        print(f"The Average AUC of the model is : {sum(kfold_auc_list) / n_splits:.4f}")
        
        # wandb.log({
        #         "avg_valid_auc" : sum(kfold_auc_list) / n_splits
        # })


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)