import cv2
import numpy as np
import imutils

# Mở camera
cap = cv2.VideoCapture(0)

# Tạo khung hình trống để lưu vết mực
canvas = None

# Biến để lưu vị trí trước đó của đối tượng
prev_center = None

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Khởi tạo khung hình canvas có cùng kích thước với khung hình hiện tại
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Làm mờ khung hình để giảm nhiễu
    blur = cv2.GaussianBlur(frame, (11, 11), 1)

    # Chuyển khung hình sang không gian màu HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Định nghĩa ngưỡng màu cho đồ vật
    lower = np.array([0, 57, 135])
    upper = np.array([199, 255, 255])

    # Tạo mặt nạ để lọc màu
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Tìm đường viền (contours) của đồ vật
    ball_cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_cnts = imutils.grab_contours(ball_cnts)

    # Nếu tìm thấy đường viền của đồ vật
    if len(ball_cnts) > 0:
        # Chọn đường viền lớn nhất
        c = max(ball_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))

        # Nếu bán kính lớn hơn một giá trị ngưỡng, vẽ lên canvas
        if radius > 10:
            if prev_center is not None:
                # Vẽ đường nối giữa vị trí trước và vị trí hiện tại
                cv2.line(canvas, prev_center, center, (255, 0, 0), 5)  # Màu xanh dương và đường dày 5 pixel
            prev_center = center
    else:
        # Đặt lại prev_center khi không phát hiện thấy đồ vật
        prev_center = None

    # Kết hợp khung hình hiện tại với khung hình canvas
    output = cv2.add(frame, canvas)

    # Hiển thị khung hình hiện tại và khung hình mask
    cv2.imshow("Tracking", output)
    cv2.imshow("Mask", mask)

    # Thoát chương trình khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
