import numpy as np
import cv2
import time


def blob_image(net, image, show_text=True):
    start = time.time()
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    end = time.time()
    if show_text:
        print("YOLO took {:.2f} seconds for process".format(end-start))
    return net, image, layerOutputs


def image_functions(image, i, trusts, boxes, COLORS, LABELS, show_text=True):
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])

    color = [int(c) for c in COLORS[IDclasses[i]]]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = "{}: {:.4f}".format(LABELS[IDclasses[i]], trusts[i])

    if show_text:
        print("> " +  text)
        print(x,y,w,h)

    cv2.putText(image, text, (x, y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, x, y, w, h


def detections(detection, boxes, trusts, IDclasses):
    scores = detection[5:]
    classeID = np.argmax(scores)
    confianca = scores[classeID]

    box = detection[0:4] * np.array([W, H, W, H])
    (centerX, centerY, width, height) = box.astype('int')

    x = int(centerX - (width/2))
    y = int(centerY - (height/2))

    boxes.append([x, y, int(width), int(height)])
    trusts.append(float(confianca))
    IDclasses.append(classeID) 

    return boxes, trusts, IDclasses


INPUT_FILE = 'inputs/input.mp4'
LABELS_FILE = 'yolo/coco.names'
CONFIG_FILE = 'yolo/yolov4.cfg'
WEIGHTS_FILE = 'yolo/yolov4.weights'

CONFIDENCE_THRESHOLD=0.5
CONFIDENCE_NMSTHRESHOLD=0.3

# Get labels class
LABELS = open(LABELS_FILE).read().strip().split("\n")

# Network connection
net = cv2.dnn.readNet(CONFIG_FILE, WEIGHTS_FILE)

# For colors (by number of labels)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Get last layers
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#####################################################################################
#####################################################################################

cap = cv2.VideoCapture(INPUT_FILE)
conected, video = cap.read() # Check connection

video_height = video.shape[0]
video_width = video.shape[1]

file_name = 'results/result.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24

output_video = cv2.VideoWriter(file_name, fourcc, fps, (video_width, video_height))

font_sm, fonte_md = 0.4, 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

while(cv2.waitKey(1) < 0):
    conected, frame = cap.read()
    if not conected:
        break
    t = time.time()
    frame = cv2.resize(frame, (video_width, video_height))
    try:
        (H, W) = frame.shape[:2]
    except:
        print('Error')
        continue

    image_cp = frame.copy()
    net, frame, layerOutputs = blob_image(net, frame)
    boxes = []
    trusts = []
    IDclasses = []

    for output in layerOutputs:
        for detection in output:
            boxes, trusts, IDclasses = detections(detection, boxes, trusts, IDclasses)
            scores = detection[5:]
            classeID = np.argmax(scores)
            trust = scores[classeID]

            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype('int')

            x = int(centerX - (width/2))
            y = int(centerY - (height/2))

            boxes.append([x, y, int(width), int(height)])
            trusts.append(float(trust))
            IDclasses.append(classeID) 

    objs = cv2.dnn.NMSBoxes(boxes, trusts, CONFIDENCE_THRESHOLD, CONFIDENCE_NMSTHRESHOLD)

    if len(objs) > 0:
        for i in objs.flatten():
            frame, x, y, w, h = image_functions(frame, i, trusts, boxes, COLORS, LABELS, show_text=False)
            objeto = image_cp[y:y + h, x:x + w]
    
    cv2.putText(frame, " frame processed in {:.2f} seconds".format(time.time()-t), (20, video_height-20), font, font_sm, (250,250,250), 0, lineType=cv2.LINE_AA)

    output_video.write(frame)

print('Ended')
output_video.release()
cv2.destroyAllWindows()
