import os
import argparse
import ctypes
from time import perf_counter

import numpy as np
from numpy.ctypeslib import ndpointer

from PIL import Image

script_dir = os.path.dirname(__file__)

def GetArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default=f'{script_dir}/inputs/sjl.jpg')
    # parser.add_argument('--kernel', type=str, default='naive')
    parser.add_argument('--radius', type=int, default=7)
    parser.add_argument('--sigma', type=float, default=10.0)

    args = parser.parse_args()

    return args

def GetDispatchFunc(libPath: str = f'{script_dir}/../build/chapter-07/libgaussian-blur.so'):
    module = ctypes.cdll.LoadLibrary(libPath)
    dispatchFunction = getattr(module, 'dispatch')
    dispatchFunction.argtypes = [
        ctypes.c_char_p,
        ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),
        ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS'),
        ctypes.c_int64,
        ctypes.c_int64,
        ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
        ctypes.c_int64
    ]
    dispatchFunction.restype = ctypes.c_double
    return dispatchFunction

def GenerateGaussianBlurringWeights(radius: int, sigma: float) -> np.ndarray:
    sideLength = radius * 2 + 1
    x = np.linspace(-radius, radius, sideLength)
    y = np.linspace(-radius, radius, sideLength)
    xv, yv = np.meshgrid(x , y)

    exponent = - (xv ** 2 + yv ** 2) / (2 * sigma ** 2)
    raw = np.exp(exponent) / (2 * np.pi * sigma ** 2)
    normalized = raw / raw.sum()
    return normalized.reshape(-1).astype(np.float32)

if __name__ == '__main__':
    args = GetArgs()

    weights = GenerateGaussianBlurringWeights(args.radius, args.sigma)

    dispatchFunc = GetDispatchFunc()

    img = Image.open(args.input)
    arr = np.array(img, dtype=np.uint8)
    height, width, __ = arr.shape

    # cpu_result = np.empty_like(arr)
    # cpu_elasped = dispatchFunc(
    #     'cpu'.encode('ascii'),
    #     cpu_result,
    #     arr,
    #     height,
    #     width,
    #     weights,
    #     args.radius
    # )
    # cpu_img = Image.fromarray(cpu_result)
    # cpu_img.save(f'{script_dir}/outputs/cpuNaive.jpg')
    # print(f'cpu saved, process time: {cpu_elasped}s')

    def RunGpu(kernelName: str):
        gpu_result = np.empty_like(arr)
        start = perf_counter()
        gpu_elasped = dispatchFunc(kernelName.encode('ascii'), gpu_result, arr, height, width, weights, args.radius)
        stop = perf_counter()
        gpu_img = Image.fromarray(gpu_result)
        gpu_img.save(f'{script_dir}/outputs/gpu_{kernelName}.jpg')
        print(f'kernelName: {kernelName}\nsaved into "outputs/gpu_{kernelName}.jpg", kernel: {gpu_elasped * 1000}ms, e2e: {(stop-start) * 1000}ms')

    RunGpu('naive')
    RunGpu('const_cache')
    RunGpu('shared')
