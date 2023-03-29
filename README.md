# iarts
A repo for my IARTS projects (AI generated art, style transfer, stable diffusion, etc.).

I've used code from amazing projects (cited below) and adapted them to my workflow in Python. Luckily enough I was able to use a large DL research server to experiment with these great models.

Code will be committed shortly. In the meantime here are some of the outputs I've generated with different models.


### Neural Neighbor Style Transfer
Code is from https://github.com/nkolkin13/NeuralNeighborStyleTransfer. Style is transferred to a single image or multiple keyframes for a video, then applied to whole sequence of images using EbSynth to generate a video.

<p align="center">
	<img width="530" height="300" src="https://github.com/K-Schubert/iarts/blob/main/media/nnst_tiger.gif">
	<img width="530" height="300" src="https://github.com/K-Schubert/iarts/blob/main/media/nnst_cat.gif">
</p>

### Stable Diffusion
Using Keras Stable Diffusion code from https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/ and https://github.com/CompVis/stable-diffusion.

<p align="center">
	<img width="250" height="250" src="https://github.com/K-Schubert/iarts/blob/main/media/budapest_vangogh_1.png">
	<img width="250" height="250" src="https://github.com/K-Schubert/iarts/blob/main/media/geneva_monet_1.png">
	<img width="250" height="250" src="https://github.com/K-Schubert/iarts/blob/main/media/geneva_munch_1.png">
	<img width="250" height="250" src="https://github.com/K-Schubert/iarts/blob/main/media/geneva_vangogh_1.png">
</p>

### Stable Diffusion Videos
The base Stable Diffusion code is from Deforum https://github.com/deforum-art. The code was refactored and updated to include motion movement in the video distributed among randomly selected keyframes, automatic prompt enhancement from a base prompt using the ´´´base_prompts.txt´´´ file and the option for single/multi GPU generation using CLI args.

<p align="center">
	<img width="250" height="250" src="https://github.com/K-Schubert/iarts/blob/main/media/stable_diff_atom.gif">
	<img width="450" height="300" src="https://github.com/K-Schubert/iarts/blob/main/media/stable_diff_video_1.gif">
	<img width="450" height="300" src="https://github.com/K-Schubert/iarts/blob/main/media/stable_diff_video_2.gif">
</p>

To run the ´´´deforum_cuda_v04_singleGPU.py´´´ file run the following commands:

´´´
virtualenv ./venv -p python3.9
source ./venv/bin/activate
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt

./venv/bin/python ./deforum_cuda_v04_singleGPU.py --device 0 --max_frames 1000 --prompt cauliflower
´´´

The ´´´device´´´ argument selects the GPU on which the model will be loaded, the ´´´max_frames´´´ argument defines how many frames in total there will be in the animation and the ´´´prompt´´´ arguments defines a base prompt which will be randomely enriched by sampling qualifiers from ´´´base_prompts.txt´´´.

### Dreambooth
Using code from https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth.

<p align="center">
	 <img width="150" height="270" src="https://github.com/K-Schubert/iarts/blob/main/media/dreambooth_1.jpg">
	 <img width="150" height="270" src="https://github.com/K-Schubert/iarts/blob/main/media/dreambooth_2.jpg">
	 <img width="150" height="270" src="https://github.com/K-Schubert/iarts/blob/main/media/dreambooth_3.jpg">
	 <img width="150" height="270" src="https://github.com/K-Schubert/iarts/blob/main/media/dreambooth_4.jpg">
	 <img width="150" height="270" src="https://github.com/K-Schubert/iarts/blob/main/media/dreambooth_5.jpg">
	 <img width="150" height="270" src="https://github.com/K-Schubert/iarts/blob/main/media/dreambooth_6.jpg">
	 <img width="150" height="270" src="https://github.com/K-Schubert/iarts/blob/main/media/dreambooth_7.jpg">
	 <img width="150" height="270" src="https://github.com/K-Schubert/iarts/blob/main/media/dreambooth_8.jpg">
</p>

### Thin-Plate-Spline Motion Animation
Using code from https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.