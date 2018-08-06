import random as pyr
import warnings
from random import randint

import numpy as np
import scipy.ndimage as ndi
from numpy import *


def autoinvert(image):
    assert amin(image) >= 0
    assert amax(image) <= 1
    if np.sum(image > 0.9) > np.sum(image < 0.1):
        return 1 - image
    else:
        return image

#
# random geometric transformations
#

def random_transform(translation=(-0.05, 0.05), rotation=(-2, 2), scale=(-0.1, 0.1), aniso=(-0.1, 0.1)):
    dx = pyr.uniform(*translation)
    dy = pyr.uniform(*translation)
    angle = pyr.uniform(*rotation)
    angle = angle * pi / 180.0
    scale = 10**pyr.uniform(*scale)
    aniso = 10**pyr.uniform(*aniso)
    return dict(angle=angle, scale=scale, aniso=aniso, translation=(dx, dy))

def transform_image(image, angle=0.0, scale=1.0, aniso=1.0, translation=(0, 0), order=1):
    dx, dy = translation
    scale = 1.0/scale
    c = cos(angle)
    s = sin(angle)
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], 'f')
    m = np.array([[c, -s], [s, c]], 'f')
    m = np.dot(sm, m)
    w, h = image.shape
    c = np.array([w, h]) / 2.0
    d = c - np.dot(m, c) + np.array([dx * w, dy * h])
    return ndi.affine_transform(image, m, offset=d, order=order, mode="nearest", output=dtype("f"))

def random_pad(image, horizontal=(0, 100)):
    l, r = np.random.randint(*horizontal, size=1), np.random.randint(*horizontal, size=1)
    return np.pad(image, ((l[0], r[0]), (0, 0)), mode="constant")
#
# random distortions
#

def bounded_gaussian_noise(shape, sigma, maxdelta):
    n, m = shape
    deltas = np.random.rand(2, n, m)
    deltas = ndi.gaussian_filter(deltas, (0, sigma, sigma))
    deltas -= np.amin(deltas)
    deltas /= np.amax(deltas)
    deltas = (2*deltas-1) * maxdelta
    return deltas

def distort_with_noise(image, deltas, order=1):
    assert deltas.shape[0] == 2
    assert image.shape == deltas.shape[1:], (image.shape, deltas.shape)
    n, m = image.shape
    xy = np.transpose(np.array(np.meshgrid(
        range(n), range(m))), axes=[0, 2, 1])
    deltas += xy
    return ndi.map_coordinates(image, deltas, order=order, mode="reflect")


def noise_distort1d(shape, sigma=100.0, magnitude=100.0):
    h, w = shape
    noise = ndi.gaussian_filter(np.random.randn(w), sigma)
    noise *= magnitude / amax(abs(noise))
    dys = array([noise]*h)
    deltas = array([dys, zeros((h, w))])
    return deltas

#
# mass preserving blur
#

def percent_black(image):
    n = prod(image.shape)
    k = sum(image < 0.5)
    return k * 100.0 / n

def binary_blur(image, sigma, noise=0.0):
    p = percent_black(image)
    blurred = ndi.gaussian_filter(image, sigma)
    if noise > 0:
        blurred += np.random.randn(*blurred.shape) * noise
    t = percentile(blurred, p)
    return array(blurred > t, 'f')

#
# multiscale noise
#

def make_noise_at_scale(shape, scale):
    h, w = shape
    h0, w0 = int(h/scale+1), int(w/scale+1)
    data = np.random.rand(h0, w0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ndi.zoom(data, scale)
    return result[:h, :w]

def make_multiscale_noise(shape, scales, weights=None, span=(0.0, 1.0)):
    if weights is None: weights = [1.0] * len(scales)
    result = make_noise_at_scale(shape, scales[0]) * weights[0]
    for s, w in zip(scales, weights):
        result += make_noise_at_scale(shape, s) * w
    lo, hi = span
    result -= amin(result)
    result /= amax(result)
    result *= (hi-lo)
    result += lo
    return result

def make_multiscale_noise_uniform(shape, srange=(1.0, 100.0), nscales=4, span=(0.0, 1.0)):
    lo, hi = log10(srange[0]), log10(srange[1])
    scales = np.random.uniform(size=nscales)
    scales = add.accumulate(scales)
    scales -= amin(scales)
    scales /= amax(scales)
    scales *= hi-lo
    scales += lo
    scales = 10**scales
    weights = 2.0 * np.random.uniform(size=nscales)
    return make_multiscale_noise(shape, scales, weights=weights, span=span)

#
# random blobs
#

def random_blobs(shape, blobdensity, size, roughness=2.0):
    from random import randint
    from builtins import range  # python2 compatible
    h, w = shape
    numblobs = int(blobdensity * w * h)
    mask = np.zeros((h, w), 'i')
    for i in range(numblobs):
        mask[randint(0, h-1), randint(0, w-1)] = 1
    dt = ndi.distance_transform_edt(1-mask)
    mask =  np.array(dt < size, 'f')
    mask = ndi.gaussian_filter(mask, size/(2*roughness))
    mask -= np.amin(mask)
    mask /= np.amax(mask)
    noise = np.random.rand(h, w)
    noise = ndi.gaussian_filter(noise, size/(2*roughness))
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return np.array(mask * noise > 0.5, 'f')

def random_blotches(image, fgblobs, bgblobs, fgscale=10, bgscale=10):
    fg = random_blobs(image.shape, fgblobs, fgscale)
    bg = random_blobs(image.shape, bgblobs, bgscale)
    return minimum(maximum(image, fg), 1-bg)

#
# random fibers
#

def make_fiber(l, a, stepsize=0.5):
    angles = np.random.standard_cauchy(l) * a
    angles[0] += 2 * pi * np.random.rand()
    angles = add.accumulate(angles)
    coss = add.accumulate(cos(angles)*stepsize)
    sins = add.accumulate(sin(angles)*stepsize)
    return array([coss, sins]).transpose(1, 0)

def make_fibrous_image(shape, nfibers=300, l=300, a=0.2, stepsize=0.5, span=(0.1, 1.0), blur=1.0):
    from builtins import range  # python2 compatible
    h, w = shape
    lo, hi = span
    result = zeros(shape)
    for i in range(nfibers):
        v = np.random.rand() * (hi-lo) + lo
        fiber = make_fiber(l, a, stepsize=stepsize)
        y, x = randint(0, h-1), randint(0, w-1)
        fiber[:, 0] += y
        fiber[:, 0] = clip(fiber[:, 0], 0, h-.1)
        fiber[:, 1] += x
        fiber[:, 1] = clip(fiber[:, 1], 0, w-.1)
        for y, x in fiber:
            result[int(y), int(x)] = v
    result = ndi.gaussian_filter(result, blur)
    result -= amin(result)
    result /= amax(result)
    result *= (hi-lo)
    result += lo
    return result


#
# print-like degradation with multiscale noise
#

def printlike_multiscale(image, blur=1.0, blotches=5e-5, inverted=None):
    if inverted:
        selector = image
    elif inverted is None:
        selector = autoinvert(image)
    else:
        selector = 1 - image

    selector = random_blotches(selector, 3*blotches, blotches)
    paper = make_multiscale_noise_uniform(image.shape, span=(0.8, 1.0))
    ink = make_multiscale_noise_uniform(image.shape, span=(0.0, 0.2))
    blurred = (ndi.gaussian_filter(selector, blur) + selector) / 2
    printed = blurred * ink + (1-blurred) * paper
    if inverted:
        return 1 - printed
    else:
        return printed


def printlike_fibrous(image, blur=1.0, blotches=5e-5, inverted=None):
    if inverted:
        selector = image
    elif inverted is None:
        selector = autoinvert(image)
    else:
        selector = 1 - image

    selector = random_blotches(selector, 3*blotches, blotches)
    paper = make_multiscale_noise(image.shape, [1.0, 5.0, 10.0, 50.0], weights=[1.0, 0.3, 0.5, 0.3], span=(0.7, 1.0))
    paper -= make_fibrous_image(image.shape, 300, 500, 0.01, span=(0.0, 0.25), blur=0.5)
    ink = make_multiscale_noise(image.shape, [1.0, 5.0, 10.0, 50.0], span=(0.0, 0.5))
    blurred = ndi.gaussian_filter(selector, blur)
    printed = blurred * ink + (1-blurred) * paper
    if inverted:
        return 1 - printed
    else:
        return printed
