from __future__ import print_function
import cv2
from cv2 import cv2 as cv
import math
import time
import argparse
from playsound import playsound
from telesign.messaging import MessagingClient


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # print("FRAMEHEIGHT: ",frameHeight)
    # print("FRAMEWIDTH: ",frameWidth)
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            # cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes



def age_detector(frame):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()

    faceProto = r"opencv_face_detector.pbtxt"
    faceModel = r"opencv_face_detector_uint8.pb"

    ageProto = r"age_deploy.prototxt"
    ageModel = r"age_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    # Load network
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    padding = 20
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No Face Detected!")
    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        # print("Age Output : {}".format(agePreds))
        # print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
        if (age != '(0-2)' and age != '(4-6)' and age != '(8-12)'):
            print("--BEEP-----BEEP-----BEEP-------BEEP-----BEEP------BEEP------BEEP--")
            # winsound.Beep(frequency, duration)
            playsound(r'Alarm.mp3')

            customer_id = "<customer_id from telesign>"
            api_key = "<api_key from telesign>"
            phone_number = "<phone_number>"
            message = "Alert, Gun Detected"
            message_type = "ARN"
            messaging = MessagingClient(customer_id, api_key)
            response = messaging.message(phone_number, message, message_type)

        label = "{}".format(age)
        # cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        # cv.imshow("Age Demo", frameFace)
    return frameFace

#
# cap1 = cv.imread(r"img33_4.jpg")
# # cap1 = cv2.VideoCapture(0)
# # # cv.waitKey()
# #
# # while True:
# #     # Capture frame-by-frame
# #     ret, frame = cap1.read()
# #
# #     dsize = (720, 360)
# #     re_img = cv.resize(frame, dsize)
# #
# #     output = age_detector(re_img)
# #     cv.imshow("Age Gender Demo", output)
# #
# #     # cv2.imshow('Video', frame)
# #
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# # cap1.release()
# # cv2.destroyAllWindows()
#
#
# print("BEFORE:")
# print('width:  ', cap1.shape[0])
# print('height: ', cap1.shape[1])
#
# # scale_percent = 50
# # #calculate the 50 percent of original dimensions
# # width = int(cap1.shape[1] * scale_percent / 200)
# # height = int(cap1.shape[0] * scale_percent / 200)
# # # dsize
# dsize = (1100, 600)
# # # print("Image size: ", dsize)
# # # resize image
# re_img = cv.resize(cap1, dsize)
# print("AFTER:")
# print('width:  ', re_img.shape[0])
# print('height: ', re_img.shape[1])
#
# output = age_detector(re_img)
# cv.imshow("Age Gender Demo", output)
# cv.waitKey()