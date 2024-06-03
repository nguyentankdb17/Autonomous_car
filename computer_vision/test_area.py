import cv2
import numpy as np

def calculate_empty_area(frame):
    # Chuyển ảnh sang thang độ xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng GaussianBlur để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Áp dụng phát hiện cạnh Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Tìm các đường viền trong ảnh
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tạo một mặt nạ rỗng có cùng kích thước với ảnh
    mask = np.zeros_like(gray)
    
    # Vẽ các đường viền lên mặt nạ
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Tính diện tích khu vực trống
    total_area = gray.shape[0] * gray.shape[1]
    filled_area = np.sum(mask == 255)
    empty_area = total_area - filled_area
    
    return empty_area, mask

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Tính diện tích trống
    empty_area, mask = calculate_empty_area(frame)
    
    # Hiển thị diện tích trống lên khung hình
    cv2.putText(frame, f'Empty Area: {empty_area}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Hiển thị khung hình và mặt nạ
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
