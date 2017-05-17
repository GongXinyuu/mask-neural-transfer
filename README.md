# Instructions

This is an advanced implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) in keras 2.0. To use this script, you need to provide a base image ,a mask of the base image and a style reference image.

### [mask_style_transfer.py](mask_style_transfer.py)

This script requires a base image ,a mask of the base image and two different style reference images. 

This script will produce a combination image which is mixed with two different styles. One is generated into the key elements of the base image, and the other is generated into the background.

## Examples
### [mask_style_transfer.py](mask_style_transfer.py)

#### Examples1

<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/img/picasso_selfport1907.jpg" width=33% height=300 alt="Picasso"> <img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/img/starry_night.jpg" width=33% height=300 alt="starry night"><img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/img/Taylor2.jpg">
<br><br> Results after 100 iterations <br>
<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/results/my05_at_iteration_100.png" width=98% height=550 alt="my05">



#### Examples2
<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/img/blue_swirls.jpg" height=300 width=33% alt="dawn sky anime"> <img src="https://github.com/GloryDream/mask-neural-transfer/blob/master/pic/img/escher_sphere.jpg?raw=true" height=300 width=33%><img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/img/Taylor2.jpg">

<br><br> Results after 100 iterations <br>

<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/results/my04_at_iteration_100.png" height=300 width=50%>

#### Example3

<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/img/picasso_selfport1907.jpg" width=33% height=300 alt="Picasso"> <img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/img/starry_night.jpg" width=33% height=300 alt="starry night"><img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/img/cat.jpg">

<br><br> Results after 100 iterations <br>

<img src="https://raw.githubusercontent.com/GloryDream/mask-neural-transfer/master/pic/results/my03_at_iteration_100.png" height=300 width=50%>



