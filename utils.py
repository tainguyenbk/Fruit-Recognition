import numpy as np
from cv2 import cv2
import random
import os


def getContour(img, kernel=(5, 5), threshold=[100, 100], iters=3, area_min=50000, fillter=4, draw=False, show=False):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgblur = cv2.medianBlur(imgray, 3)
    # imgblur = cv2.adaptiveThreshold(imgblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    # cv2.THRESH_BINARY, 11, 2)
    imgCanny = cv2.Canny(imgblur, threshold[0], threshold[1], 3, 3, True)
    # imgdialte = cv2.dilate(imgCanny,(5,5),iterations= 3)
    # imgdialte = cv2.morphologyEx(imgCanny, cv2.MORPH_OPEN, kernel)
    # imgThre = cv2.erode(imgdialte,(3,3),iterations = 1)

    if show:
        # cv2.imshow("blur", imgblur)
        # cv2.imshow("dialte",imgdialte)
        # cv2.imshow("imgThre",imgThre)
        cv2.imshow('Canny', imgCanny)
        cv2.imshow('ori', img)
    Contour, hiecherary = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Final_Contour = []
    for i in Contour:
        area = cv2.contourArea(i)
        if area > area_min:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if fillter > 0:
                if len(approx) == fillter:
                    Final_Contour.append([len(approx), area, approx, bbox, i])
            else:
                Final_Contour.append([len(approx), area, approx, bbox, i])
    Final_Contour = sorted(Final_Contour, key=lambda x: x[1], reverse=True)
    if draw:
        for con in Final_Contour:
            cv2.drawContours(img, con[4], -1, (0, 255, 255), 3)
    return img, Final_Contour


def reorder(myPoints):
    # print(myPoints)
    my_newPoints = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)

    my_newPoints[0] = myPoints[np.argmin(add)]
    my_newPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    my_newPoints[1] = myPoints[np.argmin(diff)]
    my_newPoints[2] = myPoints[np.argmax(diff)]
    my_newPoints[2][0][1] = my_newPoints[3][0][1]
    my_newPoints[1][0][0] = my_newPoints[3][0][0]
    print(my_newPoints)
    return my_newPoints


def wrapImg(img, points, w, h):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imwrap = cv2.warpPerspective(img, matrix, (w, h))
    return imwrap


def findDist(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5


def random_shadow(image):
    """
    Generates and adds random shadow
    """

    hsvImg = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsvImg[..., 2] = hsvImg[..., 2] * random.uniform(0.3, 0.6)
    return cv2.cvtColor(hsvImg, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=random.uniform(1, 5), tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


# input là img và góc quay 0-360
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# input là img và flip theo kiểu gì tra google
def flip_image(image, code):
    image = cv2.flip(image, code)
    return image


# input là img và alpha và beta muốn biết alpha beta là gì search ..............
def contras_image(image, alpha, beta):
    new_image = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

    return new_image


# nếu resize rồi thì không cần resize nữa
def resize_image(image):
    # nếu resize rồi không cần resize nữa
    image = cv2.pyrDown(image)
    image = cv2.pyrDown(image)
    return image

def foo(image):
    blank = np.zeros((100, 100, 3), dtype='uint8',)
    blank[0:100,0:100] = 255, 255, 255
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = image.shape
    roi = blank[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(image, image, mask=mask)

    result_image = cv2.add(img1_bg, img2_fg)
    return result_image




