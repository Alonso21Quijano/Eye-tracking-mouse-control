import cv2
import os.path as osp
import os
import math
import numpy as np
import time
import pyautogui


X, Y = pyautogui.size()
pyautogui.FAILSAFE = False

def weight(x, flag, width, height):
    X_1_PRED = 0.3 * width
    X_2_PRED = 0.7 * width
    X_3_PRED = 0.4 * width
    X_4_PRED = 0.6 * width
    X_CENTER = 0.5 * width

    Y_1_PRED = 0
    Y_2_PRED = 0.9 * height
    Y_CENTER = 0.5 * height

    if flag == 'x':
        if x < X_2_PRED and  (abs(x - X_1_PRED) < X_2_PRED - x):
            return 1 / (1 + (x - X_1_PRED) ** 2)
        elif x < X_CENTER and (abs(x - X_2_PRED) < X_CENTER - x):
            return 1 / (1 + (x - X_2_PRED) ** 2)
        elif x < X_3_PRED and (abs(x - X_CENTER) < X_3_PRED - x):
            return 0
        elif x < X_4_PRED and (abs(x - X_3_PRED) < X_4_PRED - x):
            return 1 / (1 + (x - X_3_PRED) ** 2)   
        else:
            return 1 / (1 + (x - X_4_PRED) ** 2)
    else:
        if x < Y_CENTER and (abs(x - Y_1_PRED) < Y_CENTER - x):
            return 1 / (1 + (x - Y_1_PRED) ** 2)
        if x < Y_2_PRED and (abs(x - Y_CENTER) < Y_2_PRED - x):
            return 0
        else:
            return 1/(1 + (x - Y_2_PRED) ** 2)

def is_max(x, ind):
    if ind < 0 or ind >= len(x):
        return False
    if ind == 0:
        return x[ind] > x[ind + 1]
    if ind == len(x):
        return x[ind] > x[ind - 1]
    return x[ind] > x[ind - 1] and x[ind] > x[ind + 1]


def find_x_s(x):
    l = len(x)
    X_1_PRED = int(0 * l)
    X_2_PRED = int(0.4 * l)
    X_3_PRED = int(0.7 * l)
    X_4_PRED = int(1 * l) - 1
    i = 0
    res = []
    while X_1_PRED + i > 0 and X_1_PRED + i < l - 1:
        if is_max(x, X_1_PRED + i):
            break
        i += 1
    res.append(X_1_PRED + i)
    i = 0
    while X_2_PRED - i > 0 and X_2_PRED - i < l - 1:
        if is_max(x, X_2_PRED - i):
            break
        i -= 1
    res.append(X_2_PRED - i)
    i = 0
    while X_3_PRED + i > 0 and X_3_PRED + i < l - 1:
        if is_max(x, X_3_PRED + i):
            break
        i += 1
    res.append(X_3_PRED + i)
    i = 0
    while X_4_PRED - i > 0 and X_4_PRED - i < l - 1:
        if is_max(x, X_4_PRED - i):
            break
        i -= 1
    res.append(X_4_PRED - i)
    return res


def find_y_s(x):
    l = len(x)
    Y_1_PRED = int(0.3 * l)
    Y_2_PRED = int(0.9 * l)

    i = 0
    res = []
    while Y_1_PRED + i > 0 and Y_1_PRED + i < l - 1:
        if is_max(x, Y_1_PRED - i):
            break
        i -= 1
    res.append(Y_1_PRED - i)
    i = 0
    while Y_2_PRED - i > 0 and Y_2_PRED - i < l - 1:
        if is_max(x, Y_2_PRED - i):
            break
        i += 1
    res.append(Y_2_PRED - i)
    return res


def get_bounds_points(eye):
    H_THRESH = 0.15
    V_THRESH = 0.1

    height, width = eye.shape

    thresh = 127
    _, eye = cv2.threshold(eye, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    one_row = np.array([1] * width)[np.newaxis]
    one_col = np.array([1] * height)[np.newaxis].T
    H = (np.sum(eye, axis=1) / width)[np.newaxis].T
    V = (np.sum(eye, axis=0) / height)[np.newaxis]
    sigmas_h = (np.sum((eye - H @ one_row) ** 2, axis=1) / width)  # horizontal projection function
    sigmas_v = (np.sum((eye - one_col @ V) ** 2, axis=0) / height)  # vertical projection function
    d_sigmas_h = np.array([((sigmas_h[i + 1] - sigmas_h[i - 1]) / 2) ** 2
                         for i in range(1, len(sigmas_h) - 1)])
    d_sigmas_v = np.array([((sigmas_v[i + 1] - sigmas_v[i - 1]) / 2) ** 2
                         for i in range(1, len(sigmas_v) - 1)])

    #d_sigmas_h /= max(d_sigmas_h)
    #d_sigmas_v /= max(d_sigmas_v)
    # sigmas_h /= max(sigmas_h)
    # sigmas_v /= max(sigmas_v)
    d_sigmas_h = np.array([val if val > H_THRESH else 0 for val in d_sigmas_h])
    d_sigmas_v = np.array([val if val > V_THRESH and val < 2 * V_THRESH else 0 for val in d_sigmas_v])

    l = len(d_sigmas_v)

    i = 1
    while (i < l - 1) and (d_sigmas_v[i] < d_sigmas_v[i - 1] or d_sigmas_v[i] < d_sigmas_v[i + 1]):
        i += 1
    left_x = i
    i = l - 2
    while (i > 0) and (d_sigmas_v[i] < d_sigmas_v[i - 1] or d_sigmas_v[i] < d_sigmas_v[i + 1]):
        i -= 1
    right_x = i

    l = len(d_sigmas_h)
    i = 1
    while (i < l - 1) and (d_sigmas_h[i] < d_sigmas_h[i - 1] or d_sigmas_h[i] < d_sigmas_h[i + 1]):
        i += 1
    up_y = i
    i = l - 2
    while (i > 0) and (d_sigmas_h[i] < d_sigmas_h[i - 1] or d_sigmas_h[i] < d_sigmas_h[i + 1]):
        i -= 1
    down_y = i

    return left_x, right_x, up_y, down_y

def sum_neighbor(pic, point, rad):
    res = 0
    count = 0
    for i in range(-rad, rad+1):
        for j in range(-rad + i, rad - i +1):
            if point[0] + i > 0 and point[0] + i < len(pic) and point[1] +j > 0 and point[1] + j < len(pic[i]):
                res += pic[point[0] + i, point[1] + j]
                count += 1
    if count:
        res /= count
    return res

def get_eye_points(pic, eye='left'):

    finder = cv2.xfeatures2d.SIFT_create(edgeThreshold=15)
    kp = finder.detect(pic, None)
    thresh = 255
    res = None
    for point in kp:
        val = sum_neighbor(pic, (math.ceil(point.pt[1]), math.ceil(point.pt[0])), 3)
        if val < thresh:
            thresh = val
            res = [point]
    name = 'left_eye' if eye == 'left' else 'right_eye'
    if res is not None and len(res):
        img = cv2.drawKeypoints(pic, res, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img = cv2.resize(img, None, img, fx=5, fy=5)

        cv2.imshow(name, img)
    return res

def spec_mean(array, now, time_const):
    times = np.array([row[0] for row in array])
    array = array[:, 1]
    coefs = np.array([array[i] * (1 - (now - times[i]) / time_const) for i in range(len(times))])
    norm_coef = np.sum([(1 - (now - times[i]) / time_const) for i in range(len(times))])

    #coefs = np.mean(array, axis=0)
    return coefs/norm_coef

def calculate_delta(left_border, right_border, up_border, down_border, center_x, center_y, kx, ky, compensate_x, compensate_y):
    dx = -center_x + (right_border - left_border) / 2
    dy = -center_y + (down_border - up_border) / 2
    return kx * dx + compensate_x,  ky * dy + compensate_y




def place_mouse(info):
    MEMORY_TIME = 0.5
    now = time.time()
    info = info[[now - row[0] < MEMORY_TIME for row in info]]
    if len(info) == 0:
        return info
    ideal = spec_mean(info, now, MEMORY_TIME)[0]
    dX, dY = calculate_delta(ideal[0], ideal[1], ideal[2], ideal[3], ideal[4], ideal[5], 800, 800, 1600, 200)
    new_x = dX + X / 2
    new_y = dY + Y / 2
    if new_x < 1:
        new_x = 1
    if new_x >= X:
        new_x = X - 1
    if new_y < 1:
        new_y = 1
    if new_y >= Y:
        new_y = Y - 1
    pyautogui.moveTo(new_x, new_y)
    return info





def main():
    #plt.ion()
    H_BOUNDARY = 0.2
    NUM = 100

    path = os.getcwd()
    face_cascade = cv2.CascadeClassifier(osp.join(path, 'haarcascades', 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(osp.join(path, 'haarcascades', 'haarcascade_eye.xml'))

    video = cv2.VideoCapture(0)

    info_packs = np.array([[0, []]])

    if not video.isOpened:
        print('Camera is not opened')
        return -1

    while True:
        left_descr = []
        right_descr = []
        ret = False
        count = 0
        while not ret and count < NUM:
            ret, pic = video.read()
        if not ret:
            video.release()
            print('Signal is lost')
            return -1
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        if len(faces):
            x, y, dx, dy = faces[0]
            for face in faces:
                 if face[0] < x:
                     x, y, dx, dy = face
            cv2.rectangle(pic, (x, y), (x + dx, y + dy), (255, 0, 0), 2)

            # left_face = pic[y:y+dy, x:x+dx]
            gray_left_face = gray[y:y + dy, x:x + dx]

            eyes = eye_cascade.detectMultiScale(gray_left_face, scaleFactor=1.1, minNeighbors=5)
            if len(eyes) > 1:
                if eyes[0][0] < eyes[1][0]:
                    lx, ly, dlx, dly = eyes[0]
                    rx, ry, drx, dry = eyes[1]
                else:
                    lx, ly, dlx, dly = eyes[1]
                    rx, ry, drx, dry = eyes[0]
            elif len(eyes) == 1:
                lx, ly, dlx, dly = eyes[0]

            if len(eyes) > 0:

                border_shift = int(H_BOUNDARY * ly)
                ly += border_shift
                dly -= 2 * border_shift

                if dly <= 1:
                    continue

                left_eye = gray_left_face[ly:ly+dly, lx:lx+dlx]
                # x1, x2, y1, y2 = get_bounds_points(left_eye)
                kps = get_eye_points(left_eye, eye='left')
                cv2.rectangle(pic, (x + lx, y + ly), (x + lx + dlx, y + ly + dly), (0, 255, 0), 2)
                #cv2.rectangle(pic, (x + lx + x1, y + ly + y1), (x + lx + x2, y +ly + y2), (255, 0, 255), 2)
                if kps is not None:
                    kp = kps[0]
                    cv2.circle(pic, (x + lx + math.ceil(kp.pt[0]), y + ly + math.ceil(kp.pt[1])), 3, (50, 50, 255))
                    # left_descr = [x1, x2, y1, y2, math.ceil(kp.pt[0]), math.ceil(kp.pt[1])]

            if len(eyes) > 1:

                border_shift = int(H_BOUNDARY * ry)
                ry += border_shift
                dry -= 2 * border_shift

                if dry <= 1:
                    continue

                right_eye = gray_left_face[ry:ry + dry, rx:rx + drx]
                kps = get_eye_points(right_eye, eye='right')
                # x1, x2, y1, y2 = get_bounds_points(right_eye)
                cv2.rectangle(pic, (x + rx, y + ry), (x + rx + drx, y + ry + dry), (0, 255, 0), 2)
                #cv2.rectangle(pic, (x + rx + x1, y + ry + y1), (x + rx + x2, y + ry + y2), (255, 0, 255), 2)
                if kps is not None:
                    kp = kps[0]
                    cv2.circle(pic, (x + rx + math.ceil(kp.pt[0]), y + ry + math.ceil(kp.pt[1])), 3, (50, 50, 255))
                    # right_descr = [x1, x2, y1, y2, math.ceil(kp.pt[0]), math.ceil(kp.pt[1])]
                '''
                if len(eyes) > 0 and len(left_descr):

                    info_packs = np.append(info_packs, np.array([[time.time(), np.array(left_descr)]]), axis=0)
                    if len(eyes) > 1 and len(right_descr):
                        info_packs[-1][1] = (info_packs[-1][1] + np.array(right_descr)) / 2.
                    info_packs = place_mouse(info_packs)
                    print('\r' + str(len(info_packs)), end="")
                    # time.sleep(1)
                '''





        cv2.imshow('analyzing video', pic)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()