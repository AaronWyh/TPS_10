import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error,roc_curve,auc,accuracy_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
from scipy.stats import norm

def features_distribution(features, data):
    print("Feature distribution of features: ")
    ncols = 5
    nrows = int((len(features)) / ncols + (len(features) % ncols > 0))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4*nrows), facecolor='#EAEAF2')

    for r in range(nrows):
        for c in range(ncols):
            col = features[r*ncols+c]
            sns.kdeplot(x=data[col], ax=axes[r, c], color='#58D68D', label='features distribution')
            axes[r, c].set_ylabel('')
            axes[r, c].set_xlabel(col, fontsize=8, fontweight='bold')
            axes[r, c].tick_params(labelsize=5, width=0.5)
            axes[r, c].xaxis.offsetText.set_fontsize(4)
            axes[r, c].yaxis.offsetText.set_fontsize(4)
    plt.show()

def missing_distribution(data):
    fig, ax = plt.subplots(figsize=(16, 6))

    bars = ax.bar(data.isna().sum().index,
                data.isna().sum().values,
                color="lightskyblue",
                edgecolor="black",
                width=0.7)
    ax.set_title("Missing feature values distribution in the train dataset", fontsize=20, pad=15)
    ax.set_ylabel("Missing values", fontsize=14, labelpad=15)
    ax.set_xlabel("Feature", fontsize=14, labelpad=10)
    ax.set_xticks([x if i%2==0 else "" for i, x in enumerate(data.columns.values)])
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.margins(0.005, 0.12)
    ax.grid(axis="y")

    plt.show()
