from torch import load
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def plot_stats(ckpt):


    checkpoint = load(ckpt)

    loss  = checkpoint['loss']
    acc_1 = checkpoint['acc_1']
    acc_2 = checkpoint['acc_2']

    batch_size = np.shape(loss[0])[0]

    plotloss = np.array([])
    plotacc_1 = np.array([])
    plotacc_2 = np.array([])

    plotloss_mean = np.array([])
    plotacc_1_mean = np.array([])
    plotacc_2_mean = np.array([])

    for i in range(13):
        plotloss = np.append(plotloss, loss[i])
        plotacc_1= np.append(plotacc_1, acc_1[i])
        plotacc_2 = np.append(plotacc_2, acc_2[i])

    for i in range(1,np.shape(plotloss)[0]):
        plotloss_mean = np.append(plotloss_mean, np.mean(plotloss[:i]))
        plotacc_1_mean= np.append(plotacc_1_mean, np.mean(plotacc_1[:i]))
        plotacc_2_mean = np.append(plotacc_2_mean, np.mean(plotacc_2[:i]))

    xvals = np.arange(np.shape(plotloss)[0])
    xvals_mean = np.arange(1,np.shape(plotloss)[0])

    fig = plt.figure()

    fig.add_subplot(3,1,1)
    plt.title("Training Loss")
    plt.xlabel("Sample #")
    plt.ylabel("Loss")
    plt.plot(xvals, plotloss, label='Loss', color='b',alpha=0.5, linewidth=0.5)
    plt.plot(xvals_mean, plotloss_mean, label='Mean Loss', color='r')
    plt.grid(True)
    plt.ylim([250,1500])
    plt.yticks([250,500,750,1000,1250,1500])
    plt.xlim([-100, xvals[-1]])
    plt.legend(loc='upper right', ncol=2, fontsize=8)


    fig.add_subplot(3,1,2)
    plt.title("Initial Depth Accuracy")
    plt.xlabel("Sample #")
    plt.ylabel("Initial Acc")
    plt.plot(xvals, plotacc_1, label='Acc', color='b',alpha=0.5, linewidth=0.5)
    plt.plot(xvals_mean, plotacc_1_mean, label='Mean Acc', color='r')
    plt.grid(True)
    plt.ylim([25,150])
    plt.yticks([25,50,75,100,125,150])
    plt.xlim([-100, xvals[-1]])
    plt.legend(loc='upper right', ncol=2, fontsize=8)


    fig.add_subplot(3,1,3)
    plt.title("Refined Depth Accuracy")
    plt.xlabel("Sample #")
    plt.ylabel("Refined Acc")
    plt.plot(xvals, plotacc_2, label='Acc', color='b',alpha=0.5, linewidth=0.5)
    plt.plot(xvals_mean, plotacc_2_mean, label='Mean Acc', color='r')
    plt.grid(True)
    plt.ylim([25,150])
    plt.xlim([-100, xvals[-1]])
    plt.yticks([25,50,75,100,125,150])
    plt.legend(loc='upper right', ncol=2, fontsize=8)

    fig.tight_layout()

    plt.show()
    

if __name__ == "__main__":

    ckpt = join('.','checkpoints','train_1647473828_13_1166')
    # chkpt = join('.','checkpoints','testresults_test_164768269_0')

    plot_stats(ckpt)