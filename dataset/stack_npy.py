'''
This script stacks all files and create a single file
'''
import numpy as np

def main():
    total = np.load('train_image1.npy')
    for i in range(2,51):
        temp = np.load('train_image' + str(i) + '.npy')
        total = np.vstack((total, temp))

    np.save('final_train.npy', total)

    total = np.load('mask_image1.npy')
    for i in range(2,51):
        temp = np.load('mask_image' + str(i) + '.npy')
        total = np.vstack((total, temp))
    np.save('final_mask.npy', total)


if __name__ == '__main__':
    main()