'''
Author       : wyx-hhhh
Date         : 2023-07-10
LastEditTime : 2023-07-12
Description  : 
'''
import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

bgsubmog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def center_point(x, y, w, h):
    return x + int(w / 2), y + int(h / 2)


def certain_area(x, y, w, h):
    if w > 50 and h > 50:
        return True
    else:
        return False


cars = []
car_num = 0
line_high = 500

while True:
    res, frame = cap.read()

    if res:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(src=frame, ksize=(5, 5), sigmaX=3)
        fgmask = bgsubmog.apply(blur)
        # 开运算
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 闭运算
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # 膨胀操作
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        # 轮廓检测
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 画一条检测线
        cv2.line(frame, (0, line_high), (frame.shape[1], line_high), (0, 255, 255), 2)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if not certain_area(x, y, w, h):
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cpoint = center_point(x, y, w, h)
            cv2.circle(frame, (cpoint), 5, (0, 0, 255), -1)
            cars.append(cpoint)

            for (x, y) in cars:
                if y > line_high - 6 and y < line_high + 6:
                    car_num += 1
                    cars.remove((x, y))
                    print(car_num)

        cv2.putText(frame, "Car Counter: " + str(car_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('video', fgmask)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()