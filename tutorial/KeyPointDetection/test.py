import cv2
import mediapipe as mp



# %%
# 获得摄像头
cap = cv2.VideoCapture(0)

# 打开摄像头
cap.open(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error")
        break
    cv2.imshow('my_window', frame)
    # 按‘q’或‘esc’退出
    if cv2.waitKey(1) in [ord('q'), 27]:
        break

# 关闭摄像头
cap.release()
# 关闭窗口
cv2.destroyAllWindows()