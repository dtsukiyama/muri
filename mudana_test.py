import os
import pytest
import shutil
import click
from click.testing import CliRunner
from yare import cpu
from muda import Scale, Noise, NoiseScale, Transform
from kantan import Scaler, Ichi, Ni, San, Both


def default_settings_model():
    '''Returns default settings for scaling function (cpu)'''
    scaler = Scale()
    settings = scaler.config()
    return settings.model

def selected_settings_model():
    scaler = Scale(model='UpConv7')
    settings = scaler.config()
    return settings.model

def scale_image():
    scaler = Scale()
    models = scaler.cpu()
    settings = scaler.config()
    transformer = Transform(models, settings)
    transformer.scale('images/small.png', 'test')
    png = os.listdir('test')[0]
    shutil.rmtree('test')
    return png

def scale_multple_images():
    scaler = Scale()
    models = scaler.cpu()
    settings = scaler.config()
    transformer = Transform(models, settings)
    transformer.scale('images', 'test')
    images = len(os.listdir('test'))
    shutil.rmtree('test')
    return images

def denoise_image():
    denoiser = Noise()
    models = denoiser.cpu()
    settings = denoiser.config()
    transformer = Transform(models, settings)
    transformer.noise('images/small.png', 'test')
    png = os.listdir('test')[0]
    shutil.rmtree('test')
    return png

def denoise_multiple_images():
    denoiser = Noise()
    models = denoiser.cpu()
    settings = denoiser.config()
    transformer = Transform(models, settings)
    transformer.noise('images', 'test')
    images = len(os.listdir('test'))
    shutil.rmtree('test')
    return images

def noise_and_scale():
    denoiser = NoiseScale()
    models = denoiser.cpu()
    settings = denoiser.config()
    transformer = Transform(models, settings)
    transformer.noise_scale('images/small.png', 'test')
    png = os.listdir('test')[0]
    shutil.rmtree('test')
    return png

def noise_and_scale_mutiple_images():
    denoiser = NoiseScale()
    models = denoiser.cpu()
    settings = denoiser.config()
    transformer = Transform(models, settings)
    transformer.noise_scale('images', 'test')
    images = len(os.listdir('test'))
    shutil.rmtree('test')
    return images

def kantan_scale():
    Scaler.go('images/small.png', 'test')
    png = os.listdir('test')[0]
    shutil.rmtree('test')
    return png

def kantan_ichi():
    Ichi.go('images/small.png', 'test')
    png = os.listdir('test')[0]
    shutil.rmtree('test')
    return png

def kantan_ni():
    Ni.go('images/small.png', 'test')
    png = os.listdir('test')[0]
    shutil.rmtree('test')
    return png

def kantan_san():
    San.go('images/small.png', 'test')
    png = os.listdir('test')[0]
    shutil.rmtree('test')
    return png

def kantan_both():
    Both.go('images/small.png', 'test')
    png = os.listdir('test')[0]
    shutil.rmtree('test')
    return png

def kantan_scale_batch():
    Scaler.go('images', 'test')
    images = len(os.listdir('test'))
    shutil.rmtree('test')
    return images

def test_default_model():
    assert default_settings_model() == 'VGG7'

def test_selected_model():
    assert selected_settings_model() == 'UpConv7'

def test_scale_image():
    assert scale_image() == 'small.png'

def test_scale_multiple_images():
    assert scale_multple_images() == 6

def test_denoiser():
    assert denoise_image() == 'small.png'

def test_multiple_denoiser_images():
    assert denoise_multiple_images() == 6

def test_noise_scale():
    assert noise_and_scale() == 'small.png'

def test_multiple_denoiser_images():
    assert noise_and_scale_mutiple_images() == 6

def test_kantan_scale():
    assert kantan_scale() == 'small.png'

def test_kantan_ichi():
    assert kantan_ichi() == 'small.png'

def test_kantan_ni():
    assert kantan_ni() == 'small.png'

def test_kantan_san():
    assert kantan_san() == 'small.png'

def test_kantan_both():
    assert kantan_both() == 'small.png'

def test_katan_scale_batch():
    assert kantan_scale_batch() == 6
