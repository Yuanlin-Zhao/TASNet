import cv2

video_path = r'E:\zyl\LGTD-main\LGTD-main\irvideo\0.MP4'
save_path = r'E:\zyl\LGTD-main\LGTD-main\qinhdaoir\test\\'

cap = cv2.VideoCapture(video_path)
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    if count < 10:
        name = '0000000' + str(count)
    if 10 <= count < 100:
        name = '000000' + str(count)
    if 100 <= count < 1000:
        name = '00000' + str(count)
    if 1000 <= count < 10000:
        name = '0000' + str(count)
    if 10000 <= count < 100000:
        name = '000' + str(count)
    cv2.imwrite(save_path + name + '.png', frame)
    print(save_path + str(count) + '.png')
    count += 1

cap.release()