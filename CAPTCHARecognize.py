from PIL import Image
import numpy as np
import os
from sklearn import neighbors
from sklearn import svm
import time

m = 3278
width, height = 20, 40
n = width * height

charCount = 4

X=np.zeros((m, n)) #img pixels matrix 20*40, total training sample count 3278
y=[] #img labels vector

ImgSize = (width,height)

RootPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')

def img2vect(imgPath):
    try:
        img = Image.open(imgPath)
        return np.asanyarray(img).flatten()
    except IOError as ex:
        print('error for {}, ignore it.{}'.format(imgPath, ex))


def prepareData(X, y, rootpath):
    print('rootpath is ', rootpath)
    count = 0
    for d in os.listdir(rootpath):
        tmp = os.path.join(os.path.join(RootPath, d), 'resized')
        for f in os.listdir(tmp):
            imgdata = img2vect(os.path.join(tmp, f))

            X[count] = imgdata
            y.append(d) 
            count +=1

def getTestData(imgPath):
    try:
        im = Image.open(imgPath)
        im = im.convert('P')
        hist = im.histogram()

        values = {}
        for i in range(256):
            values[i] = hist[i]

        temp = sorted(values.items(), key=lambda x: x[1], reverse=True)
        #print(temp[:10])
        # display the top 10 most values. like below.
        #[(225, 2597), (96, 113), (139, 104), (182, 102), (53, 69), (189, 39), (224, 27), (95, 25), (219, 25), (60, 22)]
        im2 = Image.new('P', im.size, 255)
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                pixel = im.getpixel((x,y))
                if pixel != temp[0][0]: #for character part, set it to black.
                    im2.putpixel((x,y),0)
                else:
                    im2.putpixel((x,y), 1)
        #im2.show()
        
        inletter = False
        foundletter = False
        start = 0
        end = 0
        
        #split the img to 4 chars
        letters = []
        for x in range(im2.size[0]):
            for y in range(im2.size[1]):
                pix = im2.getpixel((x,y))
                if pix != 1:
                    inletter = True
            if foundletter == False and inletter == True:
                foundletter = True
                start = x
            if foundletter == True and inletter == False:
                foundletter = False
                end = x
                letters.append((start, end))
            inletter = False
        
        #print(letters)

        count = 0
        testX = np.zeros((charCount,n))
        for letter in letters:
            im3 = im2.crop((letter[0], 0, letter[1], im2.size[1]))
            im3 = im3.resize(ImgSize)
            testX[count] = np.asarray(im3).flatten()
            count+=1

        return testX
    except Exception as ex:
        print('error: ', ex)

if __name__ =='__main__':
    prepareData(X, y, RootPath)
    
    t = time.time()
    knn = neighbors.KNeighborsClassifier()
    
    knn.fit(X, y)
    testx = getTestData('./Imgs/2.png')
    r = knn.predict(testx)

    print('knn: result is {}, time used {}sec'.format(r, time.time()-t))

    t = time.time()
    svc = svm.SVC(gamma='scale', decision_function_shape='ovo', C=1, verbose=False)

    svc.fit(X, y)
    pre = svc.predict(testx)
    print('svm: result is {}, time used {}sec'.format(pre, time.time() -t))