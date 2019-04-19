# Human-FSRCNN
Fast Super-Resolution Convolutional Neural Network for human image
This is the final project of USC course EE599-Deep learning. 
In this project, we aimed to:
  1. Train a model to built high resolution human image based on original image 
  2. Make the transform fast enough to do it in real time(20-30fps)


How to use the codes:
1. Create a folder 'dataset' in the main folder: Human-FSRCNN/dataset.
2. Create two sub-folder: 'Human-FSRCNN/dataset/HR_img_train', 'Human-FSRCNN/dataset/HR_img_test', and put test and train imgs in them(The original pics only, and the height and width must be the factor of 4).
3. run 'bash run.sh' in the Model folder.

## First version result:
generate 4* original image, create image every 100ms, PSNR=32dB, image not clear enough
![comparition](img/0_compare.jpg)
<b>
 <b>
![comparition](img/3_compare.jpg)
