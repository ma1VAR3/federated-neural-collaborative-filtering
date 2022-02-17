import numpy as np
import matplotlib.pyplot as plt


def plot(training_round):

    hits=np.load('./hit_rate.npy')

    
    plt.plot(training_round,hits,'b',label='acc')
    plt.xlabel('training_round')
    plt.ylabel('accuracy')
    plt.legend(loc="best")
    plt.savefig('acccc.png')
    
    plt.show()

arr = np.arange(0,15)
plot(arr)