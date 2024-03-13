import cv2
import keras_ocr
from sys import argv

image_filename = argv[1]

image = cv2.imread(image_filename)
image = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]

pipeline = keras_ocr.pipeline.Pipeline()

predictions = pipeline.recognize(image)

image2 = image[0].copy()

for prediction in predictions[0]:
    text = prediction[0]
    bbox = [
            (int(prediction[1][0][0]), int(prediction[1][0][1])), 
            (int(prediction[1][2][0]), int(prediction[1][2][1]))
    ]

    image2 = cv2.rectangle(image2, bbox[0], bbox[1], (255, 0, 0), 1)

image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

cv2.imshow('teste', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
