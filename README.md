# Instructions

### [weighted_neural_style_transfer.py](weighted_neural_style_transfer.py)

This is an advanced implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) in keras 2.0. To use this script, you need to provide a base image ,a mask of the base image(optional) and a style reference image.

The generated image is initialized with the base image. This algorithm uses the mask image to control the effect of gradients added on the base image, in order to make the key element in the base image to become more distinctive.

### [mask_style_transfer.py](mask_style_transfer.py)

This script requires a base image ,a mask of the base image(optional) and two different style reference images. 

This script will produce a combination image which is mixed with two different styles. One is generated into the key elements of the base image, and the other is generated into the background.

## Examples

### [weighted_neural_style_transfer.py](weighted_neural_style_transfer.py)

Base image:

![Taylor2](https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/Taylor2.jpeg)

Mask image:

![Taylor2_pascal_voc](https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/Taylor2_pascal_voc.png)

Style reference image:

![starry_night](https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/starry_night.jpg)

Result:

![maskeq0.5v0.5_at_iteration_9](https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/maskeq0.5v0.5_at_iteration_9.png)

### [mask_style_transfer.py](mask_style_transfer.py)

Base image:

![Taylor2](https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/Taylor2.jpeg)

Mask image:

![Taylor2_pascal_voc](https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/Taylor2_pascal_voc.png)

Background style reference image:

![starry_night](https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/starry_night.jpg)

Key element style reference image:

![picasso_selfport1907](https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/picasso_selfport1907.jpg)

Result:

![mix_maskv0.1_at_iteration_9](https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/mix_maskv0.1_at_iteration_9.png)

