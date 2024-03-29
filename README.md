# Muri (無理)

[![CircleCI](https://circleci.com/gh/dtsukiyama/muri.svg?style=svg)](https://circleci.com/gh/dtsukiyama/muri)

This package is based on the Muri repository which contains the Chainer implementation of waifu2x: [[2]](https://github.com/nagadomi/waifu2x); ala Tsurumeso[[1]] https://github.com/tsurumeso/waifu2x-chainer

Much of the credit should go to Tsurumeso and Nagadomi. However I just wanted to implement these models as a python module. I originally wanted to call this package 'Muda' (無駄), which means useless or a waste of time in Japanese. Amazingly there is a package called 'Muda.' I ultimately settled for 'Muri,' which means impossible.

This all started from reading Gwern's write up on [StyleGAN](https://www.gwern.net/Faces). And I was getting stuck on scaling up anime images.

This was more of a project to keep me occupied (i.e. 'muda,' a waster of time).

Here's my [LinkedIn](https://www.linkedin.com/in/david-tsukiyama-a4716b81/)


# Installation


To install from the repository source:

```
git clone https://github.com/dtsukiyama/muri.git
```

Create a virtual environment or use Anaconda:

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

Install Muri with pip:

```
cd muri
pip install .
```

# Quick Start

If you just want to scale your images there are two simple ways to go about doing so:


1. Import as a module

The easiest way to do this is:

```python
from muri.kantan import Scaler
Scaler.go('images','test')
```

You can also denoise at certain levels. For example noise level 1:

```python
from muri.kantan import Ichi
Ichi.go('images', 'test')
```

Noise level 2:

```python
from muri.kantan import Ni
Ni.go('images','test')
```

You can also do an easy default denoise and scale:

```python
from muri.kantan import Both
Both.go('images','test')
```

Caveats when using a Mac, you may get this warning:

```
UserWarning: Accelerate has been detected as a NumPy backend library.
vecLib, which is a part of Accelerate, is known not to work correctly with Chainer.
We recommend using other BLAS libraries such as OpenBLAS.
For details of the issue, please see
https://docs.chainer.org/en/stable/tips.html#mnist-example-does-not-converge-in-cpu-mode-on-mac-os-x.

Please be aware that Mac OS X is not an officially supported OS.
```

However I was able to scale images just fine on a Macbook Air. But think twice if you are looking to do hundreds of thousands or images, I am pretty sure (I hope) I will have a job before you finish doing that.

1. Command line (source)

Using cpu:

```
python muri/cpu.py --input path/to/images --output path/to/scaled/images
```

For example I have a bunch of images in the 'images' directory and I want to scale them and place them into the test directory:

```
python muri/cpu.py --input images --output test
```

If you wanted to scale a single image in the 'images' directory:

```
python muri/cpu.py --input images/sample.png --output test
```

## Using a GPU

Unfortunately I don't have any machines at home that have a gpu, my work machine did have a gpu, System76 Oryx Pro (Ahh I miss you). However I did test Muri on a GPU on Google Cloud Platform, and this perhaps is the way to go when you want to do large batches of images as you can launch an image with all Chainer dependencies and Nvidia drivers installed, it was pretty simple.

### Using GPUs on Google Cloud

You can launch a deep learning image through the [UI](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning?_ga=2.110390913.-562452682.1553380609)

However this example will be through the gcloud command line tool.

The current Chainer image is:

```
chainer-latest-cu92-experimental
```

You can find additional images [here](https://cloud.google.com/deep-learning-vm/docs/images)
And a good write up about launching deep learning images on GCP [here](https://blog.kovalevskyi.com/deep-learning-images-for-google-cloud-engine-the-definitive-guide-bc74f5fb02bc)


Create the instance:

```
gcloud compute instances create dt-training \
        --zone=us-west1-b \
        --image-family=chainer-latest-cu92-experimental\
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator='type=nvidia-tesla-k80,count=1' \
        --machine-type=n1-standard-8 \
        --boot-disk-size=120GB \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata='install-nvidia-driver=True,proxy-mode=project_editors'
```

You can also launch an instance with the supplied script, supplying the name you want to give the instance:

```
./lanuch_gpu.sh instance_name
```

In my case, e.g.:

```
./launch_gpu.sh dt-training
```


After the instance is up you should get some details about it:

```
NAME         ZONE        MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
dt-training  us-west1-b  n1-standard-8               XX.XXX.X.XX  XX.XXX.XX.XXX  RUNNING
```

If you need to you can look up your instances:

```
gcloud compute instances list
```

You can ssh into the instance, don't forget the zone:

```
gcloud compute ssh dt-training --zone=us-west1-b
```


If you need to stop the instance:

```
gcloud compute instances stop dt-training
```

To start a stopped instance:

```
gcloud compute instances start dt-training
```

And finally if you need to delete the instance:

```
gcloud compute instances delete dt-training
```

## Uploading Images

Install Muri:

```
git clone git clone https://github.com/dtsukiyama/muri.git
pip install .
```

I have a folder with over 500 256x256 pixel images, I can run the following from a local terminal to upload the images to my wm:

```
gcloud compute scp --project dt-pipeline --zone us-west1-b --recurse images_256 dt-training:/home/davidtsukiyama/muri
```

Scale all images:

```
python muri/gpu.py --input images_256 --output images_512
```

Download images from the vm to your local machine (run command from local terminal):

```
gcloud compute scp --project dt-pipeline --zone us-west1-b --recurse dt-training:/home/davidtsukiyama/muri/images_512 /path/to/folder/on/local/machine/
```

# Advanced usage

Theoretically you could initialize a model with different settings:

```python
from muri.muda import Scale, Transform
scaler = Scale()
models = scaler.cpu()
settings = scaler.config()
```

And then initialize the transformer:

```python
transformer = Transform(models, settings)
```

And finally call the method you want:

```python
transformer.scale('images/small.png', 'test/')
```


# Testing

Currently testing is set up for CPU use, not GPU use.

# Deploy Flask API on Kubernetes (WIP)

# References
- [1] tsurumeso, https://github.com/tsurumeso/waifu2x-chainer
- [2] nagadomi, "Image Super-Resolution for Anime-Style Art", https://github.com/nagadomi/waifu2x
