# ID-Card-Segmentation
Segmentation of ID Cards using U-Net

### U-Net Architecture
<img src="http://deeplearning.net/tutorial/_images/unet.jpg" width="500" height="400" alt="U-net">

### Our Result's
![Test_Image](https://github.com/AdivarekarBhumit/ID-Card-Segmentation/blob/master/images/test.jpg)
![Output_Image](https://github.com/AdivarekarBhumit/ID-Card-Segmentation/blob/master/images/output.jpg)

### Requirements
- Tensorflow-GPU 1.12
- Keras 2.1
- OpenCV 3.4.5
- Numpy 1.16

### Dataset
- Download Dataset
```
python dataset/download_dataset.py
```
- Combine To single npy file (First Download the dataset)
```
python dataset/stack_npy.py
```

### Train Model
- Start Training
```
python model/train.py
```
Training data in 100 epochs. This data was trained on google colab

### Test Model
```
python test_model.py
```

### Benchmarks
<span>
  <h4>IoU Loss</h4>
<img src="https://github.com/AdivarekarBhumit/ID-Card-Segmentation/blob/master/images/loss.svg" width="45%" height="50%" alt="IoU Loss">
  <h4>Binary Accuracy</h4>
<img src="https://github.com/AdivarekarBhumit/ID-Card-Segmentation/blob/master/images/binary_accuracy.svg" width="45%" height="50%" alt="Binary Accuracy">
</span>

<span>
  <h4>Val IoU Loss</h4>
<img src="https://github.com/AdivarekarBhumit/ID-Card-Segmentation/blob/master/images/val_loss.svg" width="45%" height="50%" alt="Val.IoU Loss">
  <h4>Val Binary Loss</h4>
<img src="https://github.com/AdivarekarBhumit/ID-Card-Segmentation/blob/master/images/val_binary_accuracy.svg" width="45%" height="50%" alt="Val.Binary Accuracy">
</span>


### Citation
Please cite this paper, if using midv dataset, link for dataset provided in paper

    @article{DBLP:journals/corr/abs-1807-05786,
      author    = {Vladimir V. Arlazarov and
                   Konstantin Bulatov and
                   Timofey S. Chernov and
                   Vladimir L. Arlazarov},
      title     = {{MIDV-500:} {A} Dataset for Identity Documents Analysis and Recognition
                   on Mobile Devices in Video Stream},
      journal   = {CoRR},
      volume    = {abs/1807.05786},
      year      = {2018},
      url       = {http://arxiv.org/abs/1807.05786},
      archivePrefix = {arXiv},
      eprint    = {1807.05786},
      timestamp = {Mon, 13 Aug 2018 16:46:35 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1807-05786},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
