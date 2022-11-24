import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob









def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if(type(v)==str):
                v = v.encode()
            txn.put(k.encode(), v)


def createDataset(outputPath,images_train, labels_train, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(images_train) == len(labels_train))
    nSamples = len(images_train)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = images_train[i]
        label = labels_train[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    # lmdb Output directory
    imagePath = '/home/public/yushilin/ocr/data/train_images'
    labelFile = "/home/public/yushilin/ocr/data/train1.txt"
    outputPath = '/home/public/yushilin/ocr/data/lmdb'
    txtLists = []
    imgPaths = []
    # Training image path, the label is in txt format, the name must be consistent with the image name, for example, the corresponding label of 123.jpg needs to be 123.txt

    '''
    imagePathList = glob.glob(path)
    print('------------', len(imagePathList), '------------')
    imgLabelLists = []
    for p in imagePathList:
        try:
            imgLabelLists.append((p, read_text(p.replace('.jpg', '.txt'))))
        except:
            continue

    # imgLabelList = [ (p,read_text(p.replace('.jpg','.txt'))) for p in imagePathList]
    ##sort by lebelList
    imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]
    '''
    with open(labelFile, "r") as f:
        data = f.readlines()
    for line in data:
        oneLine = line.split("\t")
        imgPaths.append(os.path.join(imagePath, oneLine[2]))
        txtLists.append(oneLine[3].strip())
    print(imgPaths[:6])
    print(txtLists[:6])
    createDataset(outputPath,imgPaths, txtLists)