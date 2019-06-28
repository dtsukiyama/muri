# Muri (無理)

[![CircleCI](https://circleci.com/gh/dtsukiyama/muri.svg?style=svg)](https://circleci.com/gh/dtsukiyama/muri)

This repository contains the Chainer implementation of waifu2x: [[2]](https://github.com/nagadomi/waifu2x); ala Tsurumeso[[1]] https://github.com/tsurumeso/waifu2x-chainer

Much of the credit should go to Tsurumeso and Nagadomi. However I just wanted to implement these models as a python module. I originally wanted to call this package 'Muda' (無駄), but amazingly there is a package called 'Muda.' Why call it Muda?

![](pngs/muda.png?raw=true)

Actually I wanted to call it 'Muda' because of:

![](pngs/jojo.png?raw=true)

Which is really silly. And ultimately had to settle for 'Muri,' which means impossible.

This all started from reading Gwern's write up on [StyleGAN](https://www.gwern.net/Faces). And I was getting stuck on scaling up anime images.

And Really this was more of a project to keep me occupied after suddenly being laid off from my data scientist job; this kept me from pacing back and forth in my apartment between interviews and the prospect of losing my insurance. So uhhh... yeah I am looking for work. If you are a company that:

1. Doesn't suddenly lay off their workers
2. Values people from different backgrounds and people with kids (i.e. flexible)
3. Have a data scientist opening

Here's my [LinkedIn](https://www.linkedin.com/in/david-tsukiyama-a4716b81/)

In some respects this project is not really a big deal and perhaps a waste of time, but it helped me keep me focused after losing my job; therefore not really 'Muda.'

# Installation

You can clone this repo:

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

I am working on making this a package.

# Quick Start

If you just want to scale your images there are two simple ways to go about doing so:

1. Command line

Using cpu:

```
python cpu.py --input path/to/images --output path/to/scaled/images
```

For example I have a bunch of images in the 'images' directory and I want to scale them and place them into the test directory:

```
python cpu.py --input images --output test
```

If you wanted to scale a single image in the 'images' directory:

```
python cpu.py --input images/sample.png --output test
```

2. Import as a module

The easiest way to do this is:

```python
from kantan import Scaler
Scaler.go('images','test')
```

You can also denoise at certain levels. For example noise level 1:

```python
from kantan import Ichi
Ichi.go('images', 'test')
```

Noise level 2:

```python
from kantan import Ni
Ni.go('images','test')
```

You can also do an easy default denoise and scale:

```python
from kantan import Both
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

However I was able to scale images just fine on a Macbook Air. But think twice if you are looking to do hundreds of thousands of images, I am pretty sure (I hope) I will have a job before you finish doing that.

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
python gpu.py --input images_256 --output images_512
```

Download images from the vm to your local machine (run command from local terminal):

```
gcloud compute scp --project dt-pipeline --zone us-west1-b --recurse dt-training:/home/davidtsukiyama/muri/images_512 /path/to/folder/on/loca/machine/
```

# Testing

Currently testing is set up for CPU use, not GPU use.

# Deploy Flask API on Kubernetes (WIP)

# References
------
- [1] tsurumeso, https://github.com/tsurumeso/waifu2x-chainer
- [2] nagadomi, "Image Super-Resolution for Anime-Style Art", https://github.com/nagadomi/waifu2x
