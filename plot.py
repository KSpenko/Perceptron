import numpy as np
import matplotlib.pyplot as plt

nMinibatch = 500

name = "keras_cnn_adam"

data1 = np.transpose(np.loadtxt("models/"+name+"/batch-history.dat",skiprows=1))
data2 = np.transpose(np.loadtxt("models/"+name+"/epoch-history.dat",skiprows=1))

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(15, 8)
axs[0][0].plot(data1[0], data1[1], color="cyan")
axs[0][0].plot(data2[0], data2[1])
axs[0][0].plot(data2[0], data2[3])
axs[0][0].legend(('batch_log', 'train_data','test_data'))
axs[0][0].vlines(np.argmin(data2[1])*nMinibatch, 0, 1, color='blue', linestyle='--')
axs[0][0].vlines(np.argmin(data2[3])*nMinibatch, 0, 1, color='orange', linestyle='--')
axs[0][0].set_xlabel('#Epoch')
axs[0][0].set_ylabel('Cost')

axs[0][1].loglog(data1[0], data1[1], color="cyan")
axs[0][1].loglog(data2[0], data2[1])
axs[0][1].loglog(data2[0], data2[3])
axs[0][1].legend(('batch_log', 'train_data','test_data'))
axs[0][1].vlines(np.argmin(data2[1])*nMinibatch, 0, 1, color='blue', linestyle='--')
axs[0][1].vlines(np.argmin(data2[3])*nMinibatch, 0, 1, color='orange', linestyle='--')
axs[0][1].set_xlabel('log(#Epoch)')
axs[0][1].set_ylabel('log(Cost)')

axs[1][0].plot(data1[0], data1[2], color="cyan")
axs[1][0].plot(data2[0], data2[2])
axs[1][0].plot(data2[0], data2[4])
axs[1][0].legend(('batch_log', 'train_data','test_data'))
axs[1][0].vlines(np.argmax(data2[2])*nMinibatch, 0, 1, color='blue', linestyle='--')
axs[1][0].vlines(np.argmax(data2[4])*nMinibatch, 0, 1, color='orange', linestyle='--')
axs[1][0].set_xlabel('#Epoch')
axs[1][0].set_ylabel('Accuracy')

axs[1][1].loglog(data1[0], data1[2], color="cyan")
axs[1][1].loglog(data2[0], data2[2])
axs[1][1].loglog(data2[0], data2[4])
axs[1][1].legend(('batch_log', 'train_data','test_data'))
axs[1][1].vlines(np.argmax(data2[2])*nMinibatch, 0, 1, color='blue', linestyle='--')
axs[1][1].vlines(np.argmax(data2[4])*nMinibatch, 0, 1, color='orange', linestyle='--')
axs[1][1].set_xlabel('log(#Epoch)')
axs[1][1].set_ylabel('log(Accuracy)')

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.25, hspace=0.3)
plt.savefig('models/'+name+'/logg.png')
plt.show()
