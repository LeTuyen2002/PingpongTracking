import cv2
import numpy as np
import imutils

# Mở video
cap = cv2.VideoCapture('ping_pong.mp4')

scoreA = 0 
scoreB = 0

checkA = False
checkB = False

while True:
    _, frame = cap.read()

    # Làm mờ khung hình để giảm nhiễu
    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Định nghĩa dải màu HSV để theo dõi
    lower = np.array([15, 130, 30])
    upper = np.array([85, 255, 255])

    # Tạo mặt nạ từ dải màu
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Tìm các đường viền (contours)
    ball_cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_cnts = imutils.grab_contours(ball_cnts)

    # Nếu tìm thấy đường viền
    if len(ball_cnts) > 0:
        # Lấy đường viền lớn nhất
        c = max(ball_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            # Vẽ vòng tròn xung quanh đối tượng
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    if(x < 900 and checkA == False):
        scoreB += 1
        checkA = True
        checkB = False 

    if(x > 900 and checkB == False):
        scoreA += 1
        checkA = False
        checkB = True 

    cv2.putText(img = frame, text='Score: '+ str(scoreA) + '/' + str(scoreB), org = (20,30), fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale= 1.0 ,color=(0,0,255), thickness=2)

    cv2.line(img = frame, pt1 = (900,300), pt2 = (900,700),color = (0,0,255), thickness = 4)

    # Hiển thị khung hình hiện tại  
    cv2.imshow("frame", frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
