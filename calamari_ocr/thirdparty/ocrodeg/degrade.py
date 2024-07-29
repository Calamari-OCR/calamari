import random as pyr
from random import randint

import numpy as np
import cv2 as cv


def autoinvert(image):
    assert np.amin(image) >= 0
    assert np.amax(image) <= 1
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
    angle = angle * np.pi / 180.0
    scale = 10 ** pyr.uniform(*scale)
    aniso = 10 ** pyr.uniform(*aniso)
    return dict(angle=angle, scale=scale, aniso=aniso, translation=(dx, dy))


def transform_image(image, angle=0.0, scale=1.0, aniso=1.0, translation=(0, 0), order=1):
    dx, dy = translation
    scale = 1.0 / scale
    c = np.cos(angle)
    s = np.sin(angle)
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], "f")
    m = np.array([[c, -s], [s, c]], "f")
    m = np.dot(sm, m)
    w, h = image.shape
    c = np.array([w, h]) / 2.0
    d = c - np.dot(m, c) + np.array([dx * w, dy * h])
    M = np.hstack((m, d))
    return cv.warpAffine(
        image.astype(np.dtype("f")),
        M,
        (w, h),
        flags=(cv.INTER_LINEAR),
        borderMode=cv.BORDER_REPLICATE,
    )


def random_pad(image, horizontal=(0, 100)):
    l, r = np.random.randint(*horizontal, size=1), np.random.randint(*horizontal, size=1)
    return cv.copyMakeBorder(
        image,
        l[0],
        r[0],
        0,
        0,
        cv.BORDER_CONSTANT,
        value=[0] * (1 if image.ndim == 2 else image.shape[-1]),
    )


#
# random distortions
#


def bounded_gaussian_noise(shape, sigma, maxdelta):
    n, m = shape[:2]
    deltas = np.random.rand(2, n, m)
    for n, d in enumerate(deltas):
        deltas[n] = cv.GaussianBlur(d, (0, 0), sigmaX=sigma, borderType=cv.BORDER_REFLECT)
    deltas -= np.amin(deltas)
    deltas /= np.amax(deltas)
    deltas = (2 * deltas - 1) * maxdelta
    return deltas


def distort_with_noise(image, deltas, order=1):
    assert deltas.shape[0] == 2
    assert image.shape[:2] == deltas.shape[1:], (image.shape, deltas.shape)
    n, m = image.shape[:2]
    xy = np.transpose(np.array(np.meshgrid(range(n), range(m))), axes=[0, 2, 1])
    deltas += xy
    return cv.remap(
        image,
        deltas[1].astype(np.float32),
        deltas[0].astype(np.float32),
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT,
    )


def noise_distort1d(shape, sigma=100.0, magnitude=100.0):
    h, w = shape
    noise = cv.GaussianBlur(np.random.randn(w), (0, 0), sigmaX=sigma, borderType=cv.BORDER_REFLECT)
    noise *= magnitude / np.amax(abs(noise))
    dys = np.array([noise] * h)
    deltas = np.array([dys, np.zeros((h, w))])
    return deltas


#
# mass preserving blur
#


def percent_black(image):
    n = np.prod(image.shape)
    k = sum(image < 0.5)
    return k * 100.0 / n


def binary_blur(image, sigma, noise=0.0):
    p = percent_black(image)
    blurred = cv.GaussianBlur(image, (0, 0), sigmaX=sigma, borderType=cv.BORDER_REFLECT)
    if noise > 0:
        blurred += np.random.randn(*blurred.shape) * noise
    t = np.percentile(blurred, p)
    return np.array(blurred > t, "f")


#
# multiscale noise
#


def make_noise_at_scale(shape, scale):
    h, w = shape
    h0, w0 = int(h / scale + 1), int(w / scale + 1)
    data = np.random.rand(h0, w0)
    result = cv.resize(data, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    return result[:h, :w]


def make_multiscale_noise(shape, scales, weights=None, span=(0.0, 1.0)):
    if weights is None:
        weights = [1.0] * len(scales)
    result = make_noise_at_scale(shape, scales[0]) * weights[0]
    for s, w in zip(scales, weights):
        result += make_noise_at_scale(shape, s) * w
    lo, hi = span
    result -= np.amin(result)
    result /= np.amax(result)
    result *= hi - lo
    result += lo
    return result


def make_multiscale_noise_uniform(shape, srange=(1.0, 100.0), nscales=4, span=(0.0, 1.0)):
    lo, hi = np.log10(srange[0]), np.log10(srange[1])
    scales = np.random.uniform(size=nscales)
    scales = np.add.accumulate(scales)
    scales -= np.amin(scales)
    scales /= np.amax(scales)
    scales *= hi - lo
    scales += lo
    scales = 10**scales
    weights = 2.0 * np.random.uniform(size=nscales)
    return make_multiscale_noise(shape, scales, weights=weights, span=span)


#
# random blobs
#


def random_blobs(shape, blobdensity, size, roughness=2.0):
    h, w = shape[:2]
    numblobs = max(1, int(blobdensity * w * h))  # Prevent being 0
    mask = np.zeros((h, w), np.uint8)
    for i in range(numblobs):
        mask[randint(0, h - 1), randint(0, w - 1)] = 1
    dt = cv.distanceTransform(1 - mask, cv.DIST_L2, 3)
    mask = np.array(dt < size, "f")
    mask = cv.GaussianBlur(mask, (0, 0), sigmaX=size / (2 * roughness), borderType=cv.BORDER_REFLECT)
    mask -= np.amin(mask)
    mask /= np.amax(mask)
    noise = np.random.rand(h, w)
    noise = cv.GaussianBlur(noise, (0, 0), sigmaX=size / (2 * roughness), borderType=cv.BORDER_REFLECT)
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return np.array(mask * noise > 0.5, "f")


def random_blotches(image, fgblobs, bgblobs, fgscale=10, bgscale=10):
    fg = random_blobs(image.shape[:2], fgblobs, fgscale)
    bg = random_blobs(image.shape[:2], bgblobs, bgscale)
    if image.ndim > 2:
        return np.concatenate(
            [np.minimum(np.maximum(image[..., i], fg), 1 - bg)[:, :, None] for i in range(image.shape[-1])],
            axis=image.ndim - 1,
        )
    return np.minimum(np.maximum(image, fg), 1 - bg)


#
# random fibers
#


def make_fiber(line, a, stepsize=0.5):
    angles = np.random.standard_cauchy(line) * a
    angles[0] += 2 * np.pi * np.random.rand()
    angles = np.add.accumulate(angles)
    coss = np.add.accumulate(np.cos(angles) * stepsize)
    sins = np.add.accumulate(np.sin(angles) * stepsize)
    return np.array([coss, sins]).transpose(1, 0)


def make_fibrous_image(shape, nfibers=300, le=300, a=0.2, stepsize=0.5, span=(0.1, 1.0), blur=1.0):
    h, w = shape
    lo, hi = span
    result = np.zeros(shape)
    for i in range(nfibers):
        v = np.random.rand() * (hi - lo) + lo
        fiber = make_fiber(le, a, stepsize=stepsize)
        y, x = randint(0, h - 1), randint(0, w - 1)
        fiber[:, 0] += y
        fiber[:, 0] = np.clip(fiber[:, 0], 0, h - 0.1)
        fiber[:, 1] += x
        fiber[:, 1] = np.clip(fiber[:, 1], 0, w - 0.1)
        for y, x in fiber:
            result[int(y), int(x)] = v
    result = cv.GaussianBlur(result, (0, 0), sigmaX=blur, borderType=cv.BORDER_REFLECT)
    result -= np.amin(result)
    result /= np.amax(result)
    result *= hi - lo
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

    selector = random_blotches(selector, 3 * blotches, blotches)
    paper = make_multiscale_noise_uniform(image.shape[:2], span=(0.8, 1.0))
    ink = make_multiscale_noise_uniform(image.shape[:2], span=(0.0, 0.2))
    blurred = (cv.GaussianBlur(selector, (0, 0), sigmaX=blur, borderType=cv.BORDER_REFLECT) + selector) / 2
    if blurred.ndim == 3:
        ink = np.repeat(ink[:, :, None], 3, 2)
        paper = np.repeat(paper[:, :, None], 3, 2)

    printed = blurred * ink + (1 - blurred) * paper
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

    selector = random_blotches(selector, 3 * blotches, blotches)
    paper = make_multiscale_noise(
        image.shape,
        [1.0, 5.0, 10.0, 50.0],
        weights=[1.0, 0.3, 0.5, 0.3],
        span=(0.7, 1.0),
    )
    paper -= make_fibrous_image(image.shape, 300, 500, 0.01, span=(0.0, 0.25), blur=0.5)
    ink = make_multiscale_noise(image.shape, [1.0, 5.0, 10.0, 50.0], span=(0.0, 0.5))
    blurred = cv.GaussianBlur(selector, (0, 0), sigmaX=blur, borderType=cv.BORDER_REFLECT)
    printed = blurred * ink + (1 - blurred) * paper
    if inverted:
        return 1 - printed
    else:
        return printed
