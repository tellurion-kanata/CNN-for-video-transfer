# Neural-networks-for-video-transfer
A PyTorch implementation of network in our paper Pencil Drawing Video Rendering Using Convolutional Networks.

Pencil drawing video results:
![deer](https://github.com/Kanata-Bifang/Neural-networks-for-video-transfer/blob/master/gif_result/pencil_deer.gif?raw=true)
![duck](https://github.com/Kanata-Bifang/Neural-networks-for-video-transfer/blob/master/gif_result/pencil_duck.gif?raw=true)
![dog](https://github.com/Kanata-Bifang/Neural-networks-for-video-transfer/blob/master/gif_result/pencil_dog.gif?raw=true)

You can also use it to generate videos in other styles like:
![candy](https://github.com/Kanata-Bifang/Neural-networks-for-video-transfer/blob/master/gif_result/candy.gif?raw=true)
![night](https://github.com/Kanata-Bifang/Neural-networks-for-video-transfer/blob/master/gif_result/night.gif?raw=true)

Our network is trained with loss function proposed in ReCoNet.
We have made some improvements compared with the architecture introduced in our paper, but you can still try to reproduce our result by selecting skip model to train or to test.

Our dataset can be downloaded at [mega](https://mega.nz/#!PsomzYgb!THyq95ShQT0hp_OlST3ToKu8plT3I33Zl51k-JQ0Et0) and has been updated to 640x360. 
Now we use PWC-Net to compute the ground-truth optical flow.
Videos are majorly from [videvo](https://www.videvo.net/).

If you want to train a new model, use the following command
```bash
python main.py --name [model_name] --transformer [skip/simple/skip-simple] --dataroot [training_dataset_filename] --style_image [filename]
```

If you wanto to test a model, use the following command
```bash
python main.py --name [model_name] --transformer [skip/simple/skip-simple] --dataroot [test_dataset_filename] --eval
```
The defualt transformer is skip-simple. Details about training and test settings can be found in options.

The training and evaluation dataset should be organized as:  
├─filename  
│  ├─0  
│  │  ├─frame  
│  │  ├─flow  
│  │  └─mask  
│  ├─1  
│  ...  
  
The test dataset should be organized as:  
├─filename  
│  ├─0  
│  │  └─frame   
│  ├─1  
│  ...

A checkpoint file should be orgnized as follows if you want to load a pre-trained model.  
├─checkpoints  
│  ├─[model_name]  
│  │  ├─*.pth


Reference:  
[1][ReCoNet: Real-time Coherent Video Style Transfer Network](https://arxiv.org/pdf/1807.01197.pdf)  
[2][Optical Flow Prediction with Tensorflow](https://github.com/philferriere/tfoptflow)  
[3][CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  
[4][Artistic-videos](https://github.com/manuelruder/artistic-videos)  
[5][fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style)
