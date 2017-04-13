# Instructions

### [weighted_neural_style_transfer.py](weighted_neural_style_transfer.py)

This is an advanced implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) in keras 2.0. To use this script, you need to provide a base image ,a mask of the base image(optional) and a style reference image.

The generated image is initialized with the base image. This algorithm uses the mask image to control the effect of gradients added on the base image, in order to make the key element in the base image to become more distinctive.

### [mask_style_transfer.py](mask_style_transfer.py)

This script requires a base image ,a mask of the base image and two different style reference images. 

This script will produce a combination image which is mixed with two different styles. One is generated into the key elements of the base image, and the other is generated into the background.

## Examples
### [weighted_neural_style_transfer.py](weighted_neural_style_transfer.py)
<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/Taylor2.jpeg" width=49% height=300 alt="Taylor Swift"> <img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/starry_night.jpg" width=49% height=300 alt="starry night">
<br><br> Results after 10 iterations using the INetwork<br>
<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/maskeq0.5v0.5_at_iteration_9.png" width=98% height=550 alt="blue moon lake style transfer">

### [mask_style_transfer.py](mask_style_transfer.py)
#### Examples1
<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/starry_night.jpg" height=300 width=50% alt="dawn sky anime"> <img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/picasso_selfport1907.jpg" height=300 width=49%>

<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/Taylor2.jpeg" height=300 width=50%> <img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/mix_maskv0.1_at_iteration_9.png" height=300 width=49% alt="dawn sky style transfer anime">

#### Examples2
<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/blue_swirls.jpg" height=300 width=50% alt="dawn sky anime"> <img src="https://github.com/GloryDream/mask-neural-transfer/blob/master/pic/escher_sphere.jpg?raw=true" height=300 width=49%>

<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/Taylor2.jpeg" height=300 width=50%> <img src="https://github.com/GloryDream/mask-neural-transfer/blob/master/pic/taymix2_at_iteration_9.png?raw=true" height=300 width=49% alt="dawn sky style transfer anime">



