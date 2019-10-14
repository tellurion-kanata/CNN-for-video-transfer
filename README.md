# CNN-for-video-transfer
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

Our dataset can be downloaded from [mega](https://mega.nz/#!PsomzYgb!THyq95ShQT0hp_OlST3ToKu8plT3I33Zl51k-JQ0Et0) and has been updated to 640x360. **Please do not use this dataset for any commercial purposes.**
Now we use PWC-Net to compute the ground-truth optical flow and code in [Artistic-videos](https://github.com/manuelruder/artistic-videos) to compute the occlusion mask.
Videos are majorly from [videvo](https://www.videvo.net/).

If you want to train a new model, use the following command
```bash
python main.py --name [model_name] --transformer [skip/simple/skip-simple] --dataroot [path_to_training_dataset] --style_image [filename]
```

If you wanto to test a model, use the following command
```bash
python main.py --name [model_name] --transformer [skip/simple/skip-simple] --dataroot [path_to_test_dataset] --load_epoch [epoch_name] --eval
```
The defualt transformer and epoch_name are 'skip-simple' and 'latest' respectively. Details about training and test settings can be found in options.


The training and evaluation dataset folder should be organized as:  
├─filename  
│  ├─0  
│  │  ├─frame  
│  │  ├─flow  
│  │  └─mask  
│  ├─1  
│  ...  
  
The test dataset folder should be organized as:  
├─filename  
│  ├─0  
│  │  └─frame   
│  ├─1  
│  ...

A checkpoint folder should be orgnized as follows if you want to load a pre-trained model.  
├─checkpoints  
│  ├─[model_name]  
│  │  ├─[epoch_name]_transformer_params.pth

Be free to contact me if you have any trouble using this code. My gmail is tellurion.kanata@gmail.com, or
QQ:249908966


Code Reference:  
[1][ReCoNet: Real-time Coherent Video Style Transfer Network](https://arxiv.org/pdf/1807.01197.pdf)  
[2][Optical Flow Prediction with Tensorflow](https://github.com/philferriere/tfoptflow)  
[3][CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  
[4][Artistic-videos](https://github.com/manuelruder/artistic-videos)  
[5][fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style)  
[6][flownet_pytorch](https://github.com/NVIDIA/flownet2-pytorch)  
[7][pwc_net](https://github.com/NVlabs/PWC-Net/tree/67605884f5f635e29190228e0120de08689c4375)
