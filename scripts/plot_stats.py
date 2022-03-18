from torch import load
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def plot_stats(ckpt):


    checkpoint = load(ckpt)

    loss  = checkpoint['loss']
    acc_1 = checkpoint['acc_1']
    acc_2 = checkpoint['acc_2']

    plotloss = np.array([])
    plotacc_1 = np.array([])
    plotacc_2 = np.array([])

    for i in range(13):

        mean_loss

        plotloss = np.append(plotloss, loss[i])
        plotacc_1= np.append(plotacc_1, acc_1[i])
        plotacc_2 = np.append(plotacc_2, acc_2[i])

    fig = plt.figure()

    fig.add_subplot(3,1,1)

    xvals = np.arange(np.shape(plotloss)[0])
    ax.plot(xvals, plotloss)

    plt.show()
    

if __name__ == "__main__":

    ckpt = join('.','checkpoints','train_1647473828_13_1166')
    # chkpt = join('.','checkpoints','testresults_test_164768269_0')

    plot_stats(ckpt)