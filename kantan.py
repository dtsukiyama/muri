from muda import Scale, Noise, NoiseScale, Transform

class Scaler(object):
    """I Just want to initiate default models"""
    scaler = Scale()
    scale_models = scaler.cpu()
    scale_settings = scaler.config()
    sm = Transform(scale_models, scale_settings)

    @classmethod
    def go(cls, input_path, output_path):
        cls.sm.scale(input_path, output_path)

class Ichi(object):
    noisy  = Noise()
    noise_models = noisy.cpu()
    noise_settings = noisy.config()
    nm = Transform(noise_models, noise_settings)

    @classmethod
    def go(cls, input_path, output_path):
        cls.nm.noise(input_path, output_path)

class Ni(object):
    noisy  = Noise(noise_level = 2)
    noise_models = noisy.cpu()
    noise_settings = noisy.config()
    nm = Transform(noise_models, noise_settings)

    @classmethod
    def go(cls, input_path, output_path):
        cls.nm.noise(input_path, output_path)

class San(object):
    noisy  = Noise(noise_level = 3)
    noise_models = noisy.cpu()
    noise_settings = noisy.config()
    nm = Transform(noise_models, noise_settings)

    @classmethod
    def go(cls, input_path, output_path):
        cls.nm.noise(input_path, output_path)

class Both(object):
    noise_scale = NoiseScale()
    ns_models = noise_scale.cpu()
    ns_settings = noise_scale.config()
    ns = Transform(ns_models, ns_settings)

    @classmethod
    def go(cls, input_path, output_path):
        cls.ns.noise_scale(input_path, output_path)
