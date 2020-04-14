import os
from PIL import Image, ImageFilter

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

def gaussianBlur(resizedImages, radius):
    blurImages = [
        image.filter(ImageFilter.GaussianBlur(radius)) for image in resizedImages
    ]
    return blurImages