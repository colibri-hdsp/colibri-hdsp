import pytest
from .utils import include_colibri
include_colibri()


import torch
import numpy as np
from colibri.optics.cassi import SD_CASSI, DD_CASSI, C_CASSI
from colibri.optics.spc import SPC
from colibri.optics.doe import SingleDOESpectral
from colibri.optics.sota_does import fresnel_lens, nbk7_refractive_index

@pytest.fixture
def imsize():
    b = 8
    h = 32
    w = 32
    c = 31
    return b, c, h, w

def cassi_config(imsize, mode):
    b, c, h, w = imsize

    if mode == "sd_cassi":
        out = (b, 1, h, w + c - 1)
    elif mode == "dd":
        out = (b, 1, h, w)
    elif mode == "color":
        out = (b, 1, h, w + c - 1)
    
    return out

mode_list = ["sd_cassi", "dd", "color"]

@pytest.mark.parametrize("mode", mode_list)
def test_cassi(mode, imsize):
    cube = torch.randn(imsize)
    out_shape = cassi_config(imsize, mode)

    if mode == "sd_cassi":
        cassi = SD_CASSI(imsize[1:])
    elif mode == "dd":
        cassi = DD_CASSI(imsize[1:])
    elif mode == "color":
        cassi = C_CASSI(imsize[1:])

    cube = cube.float()
    measurement = cassi(cube, type_calculation="forward")
    backward = cassi(measurement, type_calculation="backward")
    forward_backward = cassi(cube, type_calculation="forward_backward")
    assert measurement.shape == out_shape
    assert backward.shape == cube.shape
    assert forward_backward.shape == cube.shape


@pytest.fixture
def spc_config():
    img_size = [128, 32, 32]
    n_measurements = 256
    return img_size, n_measurements

def test_spc_forward(spc_config):
    img_size, n_measurements = spc_config
    spc = SPC(img_size, n_measurements)

    b, c, h, w = 1, *img_size
    x = torch.randn(b, c, h, w)

    y_forward = spc(x, type_calculation="forward")
    expected_shape = (b, n_measurements, c) 
    assert y_forward.shape == expected_shape, "Forward output shape is incorrect"



@pytest.fixture
def doe_config():
    doe_size=(100, 100)
    img_size=(32, 3, 100, 100)
    convolution_domain = "fourier" 
    type_wave_propagation = "angular_spectrum" 
    wavelengths=torch.Tensor([450, 550, 650])*1e-9
    radius_doe =  0.5e-3
    focal = 50e-3
    height_map, aperture = fresnel_lens(ny=doe_size[0], nx=doe_size[1], focal=focal, radius=radius_doe)
    source_distance = 1
    sensor_distance= 1/(1/(focal) - 1/(source_distance))
    pixel_size = (2*radius_doe)/np.min(doe_size)
    refractive_index = nbk7_refractive_index
    return doe_size, img_size, convolution_domain, type_wave_propagation, wavelengths, height_map, aperture, source_distance, sensor_distance, pixel_size, refractive_index


def test_doe_forward(doe_config):
    doe_size, img_size, convolution_domain, type_wave_propagation, wavelengths, height_map, aperture, source_distance, sensor_distance, pixel_size, refractive_index = doe_config
    images = torch.randn(img_size)

    acquisition_model = SingleDOESpectral(input_shape = img_size[1:], 
                            height_map = height_map, 
                            aperture = aperture, 
                            wavelengths = wavelengths, 
                            source_distance = source_distance, 
                            sensor_distance = sensor_distance, 
                            sensor_spectral_sensitivity = lambda x: x,
                            pixel_size = pixel_size,
                            doe_refractive_index = refractive_index,
                            approximation = type_wave_propagation,
                            domain = convolution_domain,
                            trainable = False)

    psf = acquisition_model.get_psf()
    output = acquisition_model(images)
    deconvolution = acquisition_model(output, type_calculation="backward")

    expected_shape = (len(wavelengths), *doe_size)
    assert psf.shape == expected_shape, "PSF shape is incorrect"
    assert output.shape == img_size, "Output shape is incorrect"
    assert deconvolution.shape == img_size, "Deconvolution shape is incorrect"
