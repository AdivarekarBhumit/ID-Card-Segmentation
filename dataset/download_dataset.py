'''
This script just downloads the dataset and stores the both images and their respective masks
in numpy file. There will be 50 .npy files, you can use them seperately or can run other script
to combine all of them in single file.
'''

import os
import shutil
import cv2
import json
import numpy as np 
from glob import glob

all_links = ['ftp://smartengines.com/midv-500/dataset/01_alb_id.zip',
             'ftp://smartengines.com/midv-500/dataset/02_aut_drvlic_new.zip',
             'ftp://smartengines.com/midv-500/dataset/03_aut_id_old.zip',
             'ftp://smartengines.com/midv-500/dataset/04_aut_id.zip',
             'ftp://smartengines.com/midv-500/dataset/05_aze_passport.zip', 
             'ftp://smartengines.com/midv-500/dataset/06_bra_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/07_chl_id.zip',
             'ftp://smartengines.com/midv-500/dataset/08_chn_homereturn.zip',
             'ftp://smartengines.com/midv-500/dataset/09_chn_id.zip',
             'ftp://smartengines.com/midv-500/dataset/10_cze_id.zip',
             'ftp://smartengines.com/midv-500/dataset/11_cze_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip',
             'ftp://smartengines.com/midv-500/dataset/13_deu_drvlic_old.zip',
             'ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip',
             'ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip',
             'ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip',
             'ftp://smartengines.com/midv-500/dataset/17_deu_passport_old.zip',
             'ftp://smartengines.com/midv-500/dataset/18_dza_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/19_esp_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/20_esp_id_new.zip',
             'ftp://smartengines.com/midv-500/dataset/21_esp_id_old.zip',
             'ftp://smartengines.com/midv-500/dataset/22_est_id.zip',
             'ftp://smartengines.com/midv-500/dataset/23_fin_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/24_fin_id.zip',
             'ftp://smartengines.com/midv-500/dataset/25_grc_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/26_hrv_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/27_hrv_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/28_hun_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/29_irn_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/30_ita_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/31_jpn_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/32_lva_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/33_mac_id.zip',
             'ftp://smartengines.com/midv-500/dataset/34_mda_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/35_nor_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/36_pol_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/37_prt_id.zip',
             'ftp://smartengines.com/midv-500/dataset/38_rou_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/39_rus_internalpassport.zip',
             'ftp://smartengines.com/midv-500/dataset/40_srb_id.zip',
             'ftp://smartengines.com/midv-500/dataset/41_srb_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/42_svk_id.zip',
             'ftp://smartengines.com/midv-500/dataset/43_tur_id.zip',
             'ftp://smartengines.com/midv-500/dataset/44_ukr_id.zip',
             'ftp://smartengines.com/midv-500/dataset/45_ukr_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/46_ury_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/47_usa_bordercrossing.zip',
             'ftp://smartengines.com/midv-500/dataset/48_usa_passportcard.zip',
             'ftp://smartengines.com/midv-500/dataset/49_usa_ssn82.zip',
             'ftp://smartengines.com/midv-500/dataset/50_xpo_id.zip']

def read_image(img, label):
    image = cv2.imread(img)
    mask = np.zeros(image.shape, dtype=np.uint8)
    quad = json.load(open(label, 'r'))
    coords = np.array(quad['quad'], dtype=np.int32)
    cv2.fillPoly(mask, coords.reshape(-1, 4, 2), color=(255,255,255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2))
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    mask = cv2.threshold(mask, 0,255, cv2.THRESH_BINARY)[1]
    mask = cv2.resize(mask, (256,256))
    image = cv2.resize(image, (256, 256))
    return image, mask


def main():
    i = 1

    for link in all_links:
        print('----------------------------------------------------------------------')
        print('\nDownloading:', link[40:])
        os.system('wget ' + link)
        print('Downloaded:', link[40:])
        print('Unzipping:', link[40:])
        os.system('unzip ' + link[40:])
        print('Unzipped:', link[40:].replace('.zip', ''))

        imgdir_path = './' + link[40:].replace('.zip', '') + '/images/'
        crddir_path = './' + link[40:].replace('.zip', '') + '/ground_truth/'
        # time.sleep(5)
        X = []
        Y = []
        os.remove(imgdir_path + link[40:].replace('.zip', '.tif'))
        os.remove(crddir_path + link[40:].replace('.zip', '.json'))
        for images, ground_truth in zip(sorted(os.listdir(imgdir_path)), sorted(os.listdir(crddir_path))):
            img_list = sorted(glob(imgdir_path + images + '/*.tif'))
            label_list = sorted(glob(crddir_path + ground_truth + '/*.json'))
            for img, label in zip(img_list, label_list):
                image, mask = read_image(img, label)
                X.append(image)
                Y.append(mask)

        X = np.array(X)
        Y = np.array(Y)
        Y = np.expand_dims(Y, axis=3)
        print(link[40:].replace('.zip', ''), X.shape, Y.shape)
        # print(X.shape, Y.shape)
        np.save('train_image' + str(i) + '.npy', X)
        np.save('mask_image' + str(i) + '.npy', Y)
        print('Files Saved For:', link[40:].replace('.zip', ''))
        i += 1
        print('----------------------------------------------------------------------')
        os.remove(link[40:])
        shutil.rmtree(link[40:].replace('.zip', ''))

if __name__ == '__main__':
    main()