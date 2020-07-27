import os
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

raw_folder = "../raw/"

n_subject = 0
for fn in os.listdir("../raw"):
    if fn == ".DS_Store":
        continue
    im = cv.imread(raw_folder + "/" + fn)
    im = cv.resize(im, (1000, int(im.shape[0]/im.shape[1]*1000)))

    blue_thresh = cv.adaptiveThreshold(im[:,:,0],255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,75,20)
    green_thresh = cv.adaptiveThreshold(im[:,:,1],255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,75,20)
    red_thresh = cv.adaptiveThreshold(im[:,:,2],255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,75,20)
    im = blue_thresh + green_thresh + red_thresh

    lines = cv.HoughLines(im, 1, np.pi / 180, 320, None, 0, 0)
    h_lines = []
    v_lines = []
    #xcos(t) + ysin(t) = r, a = cos(t), b = sin(t)
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)

            if (abs(b) == 0):
                v_lines.append([a, b, rho])
                continue
            if (abs(a/b) < 1):
                h_lines.append([a, b, rho])
            else:
                v_lines.append([a, b, rho])

        cross = np.zeros((len(h_lines)*len(v_lines), 2))

        i = 0
        for l1 in h_lines:
            for l2 in v_lines:
                a1 = l1[0]
                a2 = l2[0]
                b1 = l1[1]
                b2 = l2[1]
                r1 = l1[2]
                r2 = l2[2]

                if (b1 == 0):
                    x = r1/a1
                    y = (r2 - a2*x)/b2
                    cross[i] = [x,y]
                    i = i+1
                    continue

                if (a2 == 0):
                    y = r2/b2
                    x = (r1 - b1*y)/a1
                    cross[i] = [x,y]
                    i = i+1
                    continue

                x = (r1*b2 - r2*b1)/(a1*b2 - a2*b1)
                y = (r1*a2 - r2*a1)/(b1*a2 - b2*a1)
                cross[i] = [x,y]
                i = i+1

        mdl = KMeans(n_clusters=121)
        mdl.fit_predict(cross)
        centers = mdl.cluster_centers_.astype(np.int)
        ind = np.argsort(centers[:,0])
        centers = centers[ind].reshape((11,11,2))

        for i in range(11):
            indices = np.argsort(centers[i, :, 1])
            centers[i,:,:] = centers[i,indices,:]

        offset = 5
        for i in range(10):
            for j in range(10):
                try:
                    im_crop = im[centers[i,j,1]+offset+4: centers[i+1,j+1,1]-offset, centers[i,j,0] + offset+4  : centers[i+1,j+1,0]-offset-5]
                    im_crop = cv.resize(im_crop, (100,100))
                    _, im_crop = cv.threshold(im_crop,1,255,cv.THRESH_BINARY)
                    index = np.nonzero(im_crop)
                    a = [np.min(index[0]), np.max(index[0]), np.min(index[1]), np.max(index[1])]
                    im_crop = cv.resize(im[a[0]:a[1],a[2]:a[3]].astype('float32'), (100,100))
                    _, im_crop = cv.threshold(im_crop,1,255,cv.THRESH_BINARY)
                    cv.imwrite("../data_processed/" + str(n_subject) + "_" + str(i) + "_" + str(j) + ".png", im_crop)
                except:
                    print("Error at " + fn)
    n_subject = n_subject + 1

process_folder = "../data_processed/"
data = np.zeros((1900,10001))
i = 0
for fn in os.listdir(process_folder):
    if fn == ".DS_Store":
        continue
    im = cv.imread(process_folder + "/" + fn,  cv.IMREAD_UNCHANGED)
    data[i,:-1] = im.flatten()
    data[i,10000] = int(fn.split("_")[2].split(".")[0])
    i = i+1
np.save("../data",data)
