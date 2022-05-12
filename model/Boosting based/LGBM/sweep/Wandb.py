from calendar import EPOCH
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lightgbm as lgb


class CustomWandb:
    def __init__(self, args) -> None:
        self.project_name = args.project_name
        self.lr = args.learning_rate
        self.run_name = args.run_name
    
    def config(self):
        wandb.init(
        project=self.project_name,
        config={
            "lr": self.lr,
            #"dropout": random.uniform(0.01, 0.80),
            })
        wandb.run.name = self.run_name

    def set_project_name(self, project_name: str) -> None:
        '''
        project name을 설정합니다.
        '''
        self.project_name = project_name

    def set_run_name(self, run_name: str) -> None:
        self.run_name = run_name
        #wandb.run.save()
    
    def set_hpppm(self, lr: float) -> None:
        '''
        Hyper Parameter을 설정합니다.
        '''
        self.lr = lr

    
    def plot_importance(self, model, fold: int) -> None:
        '''
        각 Feature에 대한 LGB Importance를 bar plot으로 출력합니다.
        :model - LGBM model
        :fold - n번째 fold
        '''
        ax = lgb.plot_importance(model, dpi=150, figsize=(15, 7))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        
        wandb.log({f'{fold}-fold/lgb_Importance_plot': wandb.Image(ax)})
    
    
    def plot_perm_imp(self, perm_imp_df: pd.DataFrame, fold: int) -> None:
        '''
        각 Feature에 대한 Permutation Importance를 bar plot으로 출력합니다.
        :perm_imp_df - Permutation Impportance DataFrame
        :fold - n번째 fold
        '''
        plt.rcParams['figure.dpi'] = 150  # 고해상도 설정

        permutaion_importance = plt.figure(figsize=(15, 7))
        ax = permutaion_importance.add_subplot()

        ax.set_xlim(
            min(perm_imp_df["importance"]) - 0.003, max(perm_imp_df["importance"]) + 0.01
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        sns.barplot(x="importance", y="feature", data=perm_imp_df, palette="pastel")
        wandb.log({f'{fold}-fold/Permutation_Importance_plot': wandb.Image(permutaion_importance)})


    def table_perm_imp(self, perm_imp_df: pd.DataFrame, fold: int) -> None:
        '''
        각 Feature에 대한 Permutation Importance와 표준편차를 표로 출력합니다.
        :perm_imp_df - Permutation Impportance DataFrame
        :fold - n번째 fold
        '''
        table = wandb.Table(data=perm_imp_df ,columns=perm_imp_df.columns)
        wandb.log({f'{fold}-fold/Permutation_Importance_table':table}, commit=False)


    def log(self, metric):
        wandb.log(metric)

    def finish(self):
        wandb.finish()