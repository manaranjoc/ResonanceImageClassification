import os
from PIL import Image, ImageFilter
import numpy as np

def loadImages(path, dataSet, dataType):
    image_files = sorted([
        os.path.join(path,dataSet,dataType,file)
        for file in os.listdir(path+"/"+dataSet+"/"+dataType) 
            if file.endswith('jpeg')
    ])
    return image_files

def resizing(imageList, size):
    importedImages = map(Image.open,imageList)
    resizedImages = [
        image.resize(size,Image.BILINEAR) for image in importedImages
    ]
    return resizedImages

def gaussianBlur(imageList, radius):
    blurImages = [
        image.filter(ImageFilter.GaussianBlur(radius)) for image in imageList
    ]
    return blurImages

def grayReduction(imageList):
    return [
        image.convert('L') for image in imageList
    ]

def imageStandarization(image):
    pixels = np.asarray(image)
    pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels-mean)/std
    return pixels

def imagesStandarization(imageList):
    return [
        imageStandarization(image) for image in imageList
    ]

def imageNormalization(image):
    pixels = np.asarray(image)
    pixels = pixels.astype('float32')
    maximum, minimum = pixels.max(), pixels.min()
    pixels = (pixels-minimum)/(maximum-minimum)
    return pixels

def imagesNormalization(imageList):
    return [
        imageNormalization(image) for image in imageList
    ]

def arrayToImage(array):
    return Image.fromarray(array)
    