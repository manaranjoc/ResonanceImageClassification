import os

def loadImages(path, dataSet, dataType):
    image_files = sorted([
        os.path.join(path,dataSet,dataType,file)
        for file in os.listdir(path+"/"+dataSet+"/"+dataType) 
            if file.endswith('jpeg')
    ])
    return image_files