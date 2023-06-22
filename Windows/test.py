import cv2

# 读取视频文件
cap = cv2.VideoCapture('jumptest.mp4')

# 获取视频帧率和分辨率
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化视频输出
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height), isColor=True)

while cap.isOpened():
    # 逐帧读取视频
    ret, frame = cap.read()

    if ret:
        # 处理每一帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 输出处理后的每一帧
        out.write(gray)

        cv2.imshow('frame', gray)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
