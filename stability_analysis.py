import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

paths = ["./2D_100_300_12_5_distributed","./2D_100_300_12_5_localized","./3D_1000_300_15_5_distributed","./3D_1000_300_15_5_localized"]

def func(path):
    tab=np.genfromtxt(path+'/best_sols.csv',delimiter=',')

    print(tab.shape)
    mean=np.mean(tab,axis=0)
    std=np.std(tab,axis=0)

    print(mean[5],mean[10],mean[15],mean[23])
    print(std[5],std[10],std[15],std[23])

    plt.figure(figsize=(10,5),dpi=120)
    sns.barplot(y=mean,x=np.arange(1,mean.shape[0]+1))
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0.0,1.0,0.05))
    plt.xlabel('member')
    plt.ylabel('damage parameter obtained')
    plt.savefig(path+'/mean_damage.png')


if __name__ == "__main__":
    for path in paths:
        print(path)
        func(path)