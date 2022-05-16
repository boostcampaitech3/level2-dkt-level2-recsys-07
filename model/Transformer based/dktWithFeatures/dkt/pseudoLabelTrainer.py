from lightgbm import early_stopping
import numpy as np
import torch
import copy


from .dataloader import *
from .metric import *
from .optimizer import *
from .criterion import *
from .utils import *
from .scheduler import *
from .trainer import *

class Trainer:
    def __init__(self):
        pass

    def train(self, args, train_data, valid_data):
        """훈련을 마친 모델을 반환한다"""

        # args update
        self.args = args

         # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
        torch.cuda.empty_cache()
        gc.collect()

        # augmentation
        augmented_train_data = data_augmentation(train_data, args)
        if len(augmented_train_data) != len(train_data):
            print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")

        train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)
        
        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.warmup_steps = args.total_steps // 10
            
        model = get_model(args)
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        best_model = -1
        early_stopping_counter = 0
        
        print("Start Pseudo labeling")

        for epoch in range(args.n_epochs):
            
            print(f"Start Training: Epoch {epoch + 1}")

            ### TRAIN
            train_auc, train_acc, _ = train(train_loader, model, optimizer, scheduler,args)
            
            ### VALID
            valid_auc, valid_acc, preds, targets = validate(valid_loader, model, args)

            ### TODO: model save or early stopping
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_model = copy.deepcopy(model)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print(
                        f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                    )
                    break
            # scheduler
            if args.scheduler == 'plateau':
                scheduler.step(best_auc)
            else:
                scheduler.step()

        return best_model

    def evaluate(self, args, model, valid_data, is_pseudo = False):
        """
        훈련된 모델과 validation 데이터셋을 제공하면 predict 반환
        pseudo labeling 단계에서는 matric 측정을 하지 않음(is_pseudo = True)
        
        """
        pin_memory = False

        valset = DKTDataset(valid_data, args)
        valid_loader = torch.utils.data.DataLoader(valset, shuffle=False,
                                                   batch_size=args.batch_size,
                                                   pin_memory=pin_memory,
                                                   collate_fn=collate)
        if is_pseudo == False:
            auc, acc, preds, _ = validate(valid_loader, model, args)
        if is_pseudo == True: 
            auc, acc, preds, _ = validate(valid_loader, model, args, is_pseudo)
            
        return preds

    def test(self, args, model, test_data):
        return self.evaluate(args, model, test_data, is_pseudo = True)

    def get_target(self, datas):
        targets = []
        for data in datas:
            targets.append(data[1][-1])

        return np.array(targets)


class PseudoLabel:
    def __init__(self, trainer):
        self.trainer = trainer
        
        # 결과 저장용
        self.models =[]

        self.pseudo_train_data = []

    def train(self, args, train_data, valid_data):
        model = self.trainer.train(args, train_data, valid_data)

        # model 저장
        self.models.append(model)
        
        return model

    def validate(self, args, model, valid_data):
        valid_target = self.trainer.get_target(valid_data)
        valid_predict = self.trainer.evaluate(args, model, valid_data)
    
        # Metric
        valid_auc, valid_acc = get_metric(valid_target, valid_predict)

        print(f'Valid AUC : {valid_auc} Valid ACC : {valid_acc}')

    def test(self, args, model, test_data):
        test_predict = self.trainer.test(args, model, test_data)
        pseudo_labels = np.where(test_predict >= 0.5, 1, 0)
        
        return pseudo_labels

    def update_train_data(self, pseudo_labels, train_data, test_data):
        # pseudo 라벨이 담길 test 데이터 복사본
        pseudo_test_data = copy.deepcopy(test_data)

        # pseudo label 테스트 데이터 update
        for test_data, pseudo_label in zip(pseudo_test_data, pseudo_labels):
            test_data[1][-1] = pseudo_label

        # train data 업데이트
        pseudo_train_data = np.concatenate((train_data, pseudo_test_data))

        return pseudo_train_data

    def run(self, N, args, train_data, valid_data, test_data):
        """
        N은 두번째 과정을 몇번 반복할지 나타낸다.
        즉, pseudo label를 이용한 training 횟수를 가리킨다.
        """
        if N < 1:
            raise ValueError(f"N must be bigger than 1, currently {N}")

        # pseudo label training을 위한 준비 단계
        print("Preparing for pseudo label process")
        model = self.train(args, train_data, valid_data)
        self.validate(args, model, valid_data)
        pseudo_labels = self.test(args, model, test_data)
        pseudo_train_data = self.update_train_data(pseudo_labels, train_data, test_data)

        # pseudo label training 원하는 횟수만큼 반복
        for i in range(N):
            print(f'Pseudo Label Training Process {i + 1}')
            

            model = self.train(args, pseudo_train_data, valid_data)
            self.validate(args, model, valid_data)
            pseudo_labels = self.test(args, model, test_data)
            pseudo_train_data = self.update_train_data(pseudo_labels, train_data, test_data)

        self.pseudo_train_data = pseudo_train_data

    def get_pseudo_train_data(self):
        return self.pseudo_train_data