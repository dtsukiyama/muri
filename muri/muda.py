from __future__ import division
import argparse
import os
import time

import chainer
import numpy as np
from PIL import Image
import six

from muri.lib import iproc
from muri.lib import reconstruct
from muri.lib import srcnn
from muri.lib import utils

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Validate(object):

    @classmethod
    def model(cls, model):
        models = ['vgg7','upresnet10','upconv7','resnet10']
        if model.lower() not in models:
            raise ValueError('Model does not exist. Only the following models are acceptable: {}, {}, {}, {}'.format(models[0],
            models[1], models[2], models[3]))

    @classmethod
    def ns_model(cls, model):
        models = ['upresnet10','upconv7','resnet10']
        if model.lower() not in models:
            raise ValueError('Model does not exist. Only the following models are acceptable: {}, {}, {}'.format(models[0],
            models[1], models[2]))

    @classmethod
    def color(cls, color):
        colors = ['rgb', 'y']
        if color not in colors:
            raise ValueError('Only two color models available: {} and {}'.format(colors[0], colors[1]))

    @classmethod
    def noise(cls, noise_level):
        levels = [0,1,2,3]
        if noise_level not in levels:
            raise ValueError('Please set your noise level accordingly, between 0 and 3')

# scale models
class Scale(object):
    def __init__(self, model='VGG7', color='rgb',
                 tta_level=8, tta=False, batch_size=16, block_size=128, scale_ratio=2.0,
                 width=0, height=0, shorter_side=0, longer_side=0, quality=None, extension='png'):

        Validate.model(model)
        Validate.color(color)
        self.model_directory = 'muri/models/{}'.format(model.lower())
        self.models = {}
        self.channel = 3 if color == 'rgb' else 1
        self.color = color
        self.model = model
        self.tta_level = tta_level
        self.tta = tta
        self.batch_size = batch_size
        self.block_size = block_size
        self.scale_ratio = scale_ratio
        self.width = width
        self.height = height
        self.shorter_side = shorter_side
        self.longer_side = longer_side
        self.quality = quality
        self.extension = extension

    def config(self):
        settings = Namespace()
        settings.model_dir = self.model_directory
        settings.color = self.color
        settings.model = self.model
        settings.ch = self.channel
        settings.tta_level = self.tta_level
        settings.batch_size = self.batch_size
        settings.block_size = self.block_size
        settings.scale_ratio = self.scale_ratio
        settings.height = self.height
        settings.width = self.width
        settings.shorter_side = self.shorter_side
        settings.longer_side = self.longer_side
        settings.tta = self.tta
        settings.quality = self.quality
        settings.extension = self.extension
        return settings

    def cpu(self):
        model_name = 'anime_style_scale_{}.npz'.format(self.color)
        model_path = os.path.join(self.model_directory, model_name)
        self.models['scale'] = srcnn.archs[self.model](self.channel)
        chainer.serializers.load_npz(model_path, self.models['scale'])
        return self.models

    def gpu(self, n_gpu):
        model_name = 'anime_style_scale_{}.npz'.format(self.color)
        model_path = os.path.join(self.model_directory, model_name)
        self.models['scale'] = srcnn.archs[self.model](self.channel)
        chainer.serializers.load_npz(model_path, self.models['scale'])
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(n_gpu).use()
        for _, model in self.models.items():
            model.to_gpu()
        return self.models

class Noise(object):
    def __init__(self, model='VGG7', noise_level = 1, color='rgb',
                 tta_level=8, tta=False, batch_size=16, block_size=128, scale_ratio=2.0,
                 width=0, height=0, shorter_side=0, longer_side=0, quality=None, extension='png'):

        Validate.model(model)
        Validate.color(color)
        Validate.noise(noise_level)
        self.model_directory = 'muri/models/{}'.format(model.lower())
        self.models = {}
        self.channel = 3 if color == 'rgb' else 1
        self.color = color
        self.noise_level = noise_level
        self.model = model
        self.tta_level = tta_level
        self.tta = tta
        self.batch_size = batch_size
        self.block_size = block_size
        self.scale_ratio = scale_ratio
        self.width = width
        self.height = height
        self.shorter_side = shorter_side
        self.longer_side = longer_side
        self.quality = quality
        self.extension = extension

    def config(self):
        settings = Namespace()
        settings.model_dir = self.model_directory
        settings.color = self.color
        settings.noise_level = self.noise_level
        settings.model = self.model
        settings.ch = self.channel
        settings.tta_level = self.tta_level
        settings.batch_size = self.batch_size
        settings.block_size = self.block_size
        settings.scale_ratio = self.scale_ratio
        settings.height = self.height
        settings.width = self.width
        settings.shorter_side = self.shorter_side
        settings.longer_side = self.longer_side
        settings.tta = self.tta
        settings.quality = self.quality
        settings.extension = self.extension
        return settings

    def cpu(self):
        model_name = 'anime_style_noise{}_{}.npz'.format(self.noise_level, self.color)
        model_path = os.path.join(self.model_directory, model_name)
        if not os.path.exists(model_path):
            model_name = 'anime_style_noise{}_scale_{}.npz'.format(self.noise_level, self.color)
            model_path = os.path.join(self.model_directory, model_name)
        self.models['noise'] = srcnn.archs[self.model](self.channel)
        chainer.serializers.load_npz(model_path, self.models['noise'])
        return self.models

    def gpu(self, n_gpu):
        model_name = 'anime_style_noise{}_{}.npz'.format(self.noise_level, self.color)
        model_path = os.path.join(self.model_directory, model_name)
        if not os.path.exists(model_path):
            model_name = 'anime_style_noise{}_scale_{}.npz'.format(self.noise_level, self.color)
            model_path = os.path.join(self.model_directory, model_name)
        self.models['noise'] = srcnn.archs[self.model](self.channel)
        chainer.serializers.load_npz(model_path, self.models['noise'])
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(n_gpu).use()
        for _, model in self.models.items():
            model.to_gpu()
        return self.models

class NoiseScale(object):
    def __init__(self, model='UpConv7', noise_level=1, color='rgb',
                 tta_level=8, tta=False, batch_size=16, block_size=128, scale_ratio=2.0,
                 width=0, height=0, shorter_side=0, longer_side=0, quality=None, extension='png'):

        Validate.ns_model(model)
        Validate.color(color)
        Validate.noise(noise_level)
        self.model_directory = 'muri/models/{}'.format(model.lower())
        self.models = {}
        self.channel = 3 if color == 'rgb' else 1
        self.color = color
        self.noise_level = noise_level
        self.model = model
        self.tta_level = tta_level
        self.tta = tta
        self.batch_size = batch_size
        self.block_size = block_size
        self.scale_ratio = scale_ratio
        self.width = width
        self.height = height
        self.shorter_side = shorter_side
        self.longer_side = longer_side
        self.quality = quality
        self.extension = extension

    def config(self):
        settings = Namespace()
        settings.model_dir = self.model_directory
        settings.color = self.color
        settings.noise_level = self.noise_level
        settings.model = self.model
        settings.ch = self.channel
        settings.tta_level = self.tta_level
        settings.batch_size = self.batch_size
        settings.block_size = self.block_size
        settings.scale_ratio = self.scale_ratio
        settings.height = self.height
        settings.width = self.width
        settings.shorter_side = self.shorter_side
        settings.longer_side = self.longer_side
        settings.tta = self.tta
        settings.quality = self.quality
        settings.extension = self.extension
        return settings

    def cpu(self):
        model_name = 'anime_style_noise{}_scale_{}.npz'.format(self.noise_level, self.color)
        model_path = os.path.join(self.model_directory, model_name)
        if os.path.exists(model_path):
            self.models['noise_scale'] = srcnn.archs[self.model](self.channel)
            chainer.serializers.load_npz(model_path, self.models['noise_scale'])
            alpha_model_name = 'anime_style_scale_{}.npz'.format(self.color)
            alpha_model_path = os.path.join(self.model_directory, alpha_model_name)
            self.models['alpha'] = srcnn.archs[self.model](self.channel)
            chainer.serializers.load_npz(alpha_model_path, self.models['alpha'])
        return self.models

    def gpu(self, n_gpu):
        model_name = 'anime_style_noise{}_{}.npz'.format(self.noise_level, self.color)
        model_path = os.path.join(self.model_directory, model_name)
        if os.path.exists(model_path):
             self.models['noise_scale'] = srcnn.archs[self.model](self.channel)
             chainer.serializers.load_npz(model_path, self.models['noise_scale'])
             alpha_model_name = 'anime_style_scale_{}.npz'.format(self.color)
             alpha_model_path = os.path.join(self.model_directory, alpha_model_name)
             self.models['alpha'] = srcnn.archs[self.model](self.channel)
             chainer.serializers.load_npz(alpha_model_path, self.models['alpha'])
             chainer.serializers.load_npz(model_path, models['noise'])
             chainer.backends.cuda.check_cuda_available()
             chainer.backends.cuda.get_device(n_gpu).use()
             for _, model in self.models.items():
                 model.to_gpu()
        return self.models

class Transform(object):
    def __init__(self, models, settings, extension='png'):
        self.models = models
        self.input_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']
        self.output_exts = ['.png', '.webp']
        self.settings = settings

    def denoise_image(self, src, model):
        dst, alpha = self.split_alpha(src, model)
        six.print_('Level {} denoising...'.format(self.settings.noise_level), end=' ', flush=True)
        if self.settings.tta:
            dst = reconstruct.image_tta(
                dst, model, self.settings.tta_level,
                            self.settings.block_size,
                            self.settings.batch_size)
        else:
            dst = reconstruct.image(dst, model, self.settings.block_size,
                                                self.settings.batch_size)
        if model.inner_scale != 1:
            dst = dst.resize((src.size[0], src.size[1]), Image.LANCZOS)
        six.print_('OK')
        if alpha is not None:
            dst.putalpha(alpha)
        return dst

    def split_alpha(self, src, model):
        alpha = None
        if src.mode in ('L', 'RGB', 'P'):
            if isinstance(src.info.get('transparency'), bytes):
                src = src.convert('RGBA')
        rgb = src.convert('RGB')
        if src.mode in ('LA', 'RGBA'):
            six.print_('Splitting alpha channel...', end=' ', flush=True)
            alpha = src.split()[-1]
            rgb = iproc.alpha_make_border(rgb, alpha, model)
            six.print_('OK')
        return rgb, alpha

    def upscale_image(self, src, scale_model, alpha_model=None):
        dst, alpha = self.split_alpha(src, scale_model)
        for i in range(int(np.ceil(np.log2(self.settings.scale_ratio)))):
            six.print_('2.0x upscaling...', end=' ', flush=True)
            model = scale_model if i == 0 or alpha_model is None else alpha_model
            if model.inner_scale == 1:
                dst = iproc.nn_scaling(dst, 2)  # Nearest neighbor 2x scaling
                alpha = iproc.nn_scaling(alpha, 2)  # Nearest neighbor 2x scaling
            if self.settings.tta:
                dst = reconstruct.image_tta(
                    dst, model, self.settings.tta_level,
                                self.settings.block_size,
                                self.settings.batch_size)
            else:
                dst = reconstruct.image(dst, model,
                                             self.settings.block_size,
                                             self.settings.batch_size)
            if alpha_model is None:
                alpha = reconstruct.image(
                    alpha, scale_model, self.settings.block_size, self.settings.batch_size)
            else:
                alpha = reconstruct.image(
                    alpha, alpha_model, self.settings.block_size, self.settings.batch_size)
            six.print_('OK')
        dst_w = int(np.round(src.size[0] * self.settings.scale_ratio))
        dst_h = int(np.round(src.size[1] * self.settings.scale_ratio))
        if dst_w != dst.size[0] or dst_h != dst.size[1]:
            six.print_('Resizing...', end=' ', flush=True)
            dst = dst.resize((dst_w, dst_h), Image.LANCZOS)
            six.print_('OK')
        if alpha is not None:
            if alpha.size[0] != dst_w or alpha.size[1] != dst_h:
                alpha = alpha.resize((dst_w, dst_h), Image.LANCZOS)
            dst.putalpha(alpha)
        return dst

    def build_filelist(self, input, output):
        outname = None
        outdir = output
        outext = '.' + self.settings.extension
        if os.path.isdir(input):
            filelist = utils.load_filelist(input)
        else:
            tmpname, tmpext = os.path.splitext(os.path.basename(output))
            if tmpext in self.output_exts:
                outext = tmpext
                outname = tmpname
                outdir = os.path.dirname(output)
                outdir = './' if outdir == '' else outdir
            elif not tmpext == '':
                raise ValueError('Format {} is not supported'.format(tmpext))
            filelist = [input]
        # create outpur directory
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        return outext, outname, outdir, filelist

    def process_image(self, outext, outname, outdir, filelist):
        for path in filelist:
            if outname is None or len(filelist) > 1:
                outname, outext = os.path.splitext(os.path.basename(path))
            outpath = os.path.join(outdir, '{}{}'.format(outname, outext))
            if outext.lower() in self.input_exts:
                src = Image.open(path)
                w, h = src.size[:2]
                if self.settings.width != 0:
                    self.settings.scale_ratio = self.settings.width / w
                elif self.settings.height != 0:
                    self.settings.scale_ratio = self.settings.height / h
                elif self.settings.shorter_side != 0:
                    if w < h:
                        self.settings.scale_ratio = self.settings.shorter_side / w
                    else:
                        self.settings.scale_ratio = self.settings.shorter_side / h
                elif self.settings.longer_side != 0:
                    if w > h:
                        self.settings.scale_ratio = self.settings.longer_side / w
                    else:
                        self.settings.scale_ratio = self.settings.longer_side / h
                dst = src.copy()
                outname += '_(tta{})'.format(self.settings.tta_level) if self.settings.tta else '_'
        return outdir, outpath, outname, dst, src

    def process_collection(self, outext, outname, outdir, filelist):
        collection = []
        for path in filelist:
            if outname is None or len(filelist) > 1:
                outname, outext = os.path.splitext(os.path.basename(path))
            outpath = os.path.join(outdir, '{}{}'.format(outname, outext))
            if outext.lower() in self.input_exts:
                src = Image.open(path)
                w, h = src.size[:2]
                if self.settings.width != 0:
                    self.settings.scale_ratio = self.settings.width / w
                elif self.settings.height != 0:
                    self.settings.scale_ratio = self.settings.height / h
                elif self.settings.shorter_side != 0:
                    if w < h:
                        self.settings.scale_ratio = self.settings.shorter_side / w
                    else:
                        self.settings.scale_ratio = self.settings.shorter_side / h
                elif self.settings.longer_side != 0:
                    if w > h:
                        self.settings.scale_ratio = self.settings.longer_side / w
                    else:
                        self.settings.scale_ratio = self.settings.longer_side / h
                dst = src.copy()
                outname += '_(tta{})'.format(self.settings.tta_level) if self.settings.tta else '_'
                collection.append((outdir, outpath, outname, dst, src))
        return collection

    def scale(self, input, output):
        outext, outname, outdir, filelist = self.build_filelist(input, output)
        collection = self.process_collection(outext, outname, outdir, filelist)

        for features in collection:
            outdir  = features[0]
            outpath = features[1]
            outname = features[2]
            dst     = features[3]
            src     = features[4]

            outname += '(scale{:.1f}x)'.format(self.settings.scale_ratio)
            dst = self.upscale_image(dst, self.models['scale'])
            outname += '({}_{}){}'.format(self.settings.model, self.settings.color, outext)

            if os.path.exists(outpath):
                outpath = os.path.join(outdir, outname)

            lossless = self.settings.quality is None
            quality = 100 if lossless else self.settings.quality
            icc_profile = src.info.get('icc_profile')
            icc_profile = "" if icc_profile is None else icc_profile
            dst.convert(src.mode).save(outpath, quality=quality, lossless=lossless, icc_profile=icc_profile)
            six.print_('Saved as \'{}\''.format(outpath))

    def noise(self, input, output):
        outext, outname, outdir, filelist = self.build_filelist(input, output)
        collection = self.process_collection(outext, outname, outdir, filelist)
        for features in collection:
            outdir  = features[0]
            outpath = features[1]
            outname = features[2]
            dst     = features[3]
            src     = features[4]

            outname += '(noise{})'.format(self.settings.noise_level)
            dst = self.denoise_image(dst, self.models['noise'])
            outname += '({}_{}){}'.format(self.settings.model, self.settings.color, outext)
            if os.path.exists(outpath):
               outpath = os.path.join(outdir, outname)

            lossless = self.settings.quality is None
            quality = 100 if lossless else self.settings.quality
            icc_profile = src.info.get('icc_profile')
            icc_profile = "" if icc_profile is None else icc_profile
            dst.convert(src.mode).save(outpath, quality=quality, lossless=lossless, icc_profile=icc_profile)
            six.print_('Saved as \'{}\''.format(outpath))

    def noise_scale(self, input, output):
        outext, outname, outdir, filelist = self.build_filelist(input, output)
        collection = self.process_collection(outext, outname, outdir, filelist)
        for features in collection:
            outdir  = features[0]
            outpath = features[1]
            outname = features[2]
            dst     = features[3]
            src     = features[4]

            outname += '(noise{}_scale{:.1f}x)'.format(self.settings.noise_level,
                                                       self.settings.scale_ratio)

            dst = self.upscale_image(dst, self.models['noise_scale'], self.models['alpha'])
            outname += '({}_{}){}'.format(self.settings.model, self.settings.color, outext)
            if os.path.exists(outpath):
                outpath = os.path.join(outdir, outname)

            lossless = self.settings.quality is None
            quality = 100 if lossless else self.settings.quality
            icc_profile = src.info.get('icc_profile')
            icc_profile = "" if icc_profile is None else icc_profile
            dst.convert(src.mode).save(outpath, quality=quality, lossless=lossless, icc_profile=icc_profile)
            six.print_('Saved as \'{}\''.format(outpath))
