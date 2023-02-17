# iarts
A repo for my IARTS projects (AI generated art, style transfer, stable diffusion, etc.).

I've used code from amazing projects (cited below) and adapted them to my workflow in Python. Luckily enough I was able to use a large DL research server to experiment with these great models.

Code will be committed shortly. In the meantime here are some of the outputs I've generated with different models.


### Neural Neighbor Style Transfer
Code is from https://github.com/nkolkin13/NeuralNeighborStyleTransfer. Style is transferred to a single image or multiple keyframes for a video, then applied to whole sequence of images using EbSynth to generate a video.

![NNST applied to a tiger](./media/nnst_tiger.gif)
![NNST applied to a cat](./media/nnst_cat.gif)

### Stable Diffusion
Using Keras Stable Diffusion code from https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/ and https://github.com/CompVis/stable-diffusion.

![Budapest in starry night style](./media/budapest_vangogh_1.pdf)
![Geneva in Monet style](./media/budapest_monet_1.pdf)
![Geneva in Munch style](./media/budapest_munch_1.pdf)
![Geneva in starry night style](./media/budapest_vangogh_1.pdf)

### Stable Diffusion Videos
Using code from Deforum https://github.com/deforum-art.

![The journey of an atom with stable diffusion](./media/stable_diff_atom.gif)
![AI dreams #1](./media/stable_diff_video_1.gif)
![AI dreams #2](./media/stable_diff_video_2.gif)

### Dreambooth
Using code from https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth.

![Dreambooth 1](./media/dreambooth_1.pdf)
![Dreambooth 2](./media/dreambooth_2.pdf)
![Dreambooth 3](./media/dreambooth_3.pdf)
![Dreambooth 4](./media/dreambooth_4.pdf)
![Dreambooth 5](./media/dreambooth_5.pdf)
![Dreambooth 6](./media/dreambooth_6.pdf)
![Dreambooth 7](./media/dreambooth_7.pdf)
![Dreambooth 8](./media/dreambooth_8.pdf)

### Thin-Plate-Spline Motion Animation
Using code from https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.