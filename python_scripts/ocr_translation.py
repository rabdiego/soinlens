import cv2
import keras_ocr
import numpy as np
from sys import argv

def are_all_clusterized(list):
    for element in list:
        if element < 0:
            return False
    return True


def get_min_size_from_image(image):
    img_shape = np.array(image).shape
    width = img_shape[1]
    height = img_shape[2]
    return width if width < height else height


def get_bbox_and_text_from_raw_image(image):
    # Instanciate objects
    pipeline = keras_ocr.pipeline.Pipeline()
    wois = list()  # Word On Image

    # Get predictions
    predictions = pipeline.recognize(image)

    # Appending WOIs to list
    for prediction in predictions[0]:
        text = prediction[0]
        bbox = [
                (int(prediction[1][0][0]), int(prediction[1][0][1])), 
                (int(prediction[1][2][0]), int(prediction[1][2][1]))
        ]

        wois.append((text, bbox))
    
    return wois


def clusterize_into_paragraphs(wois, size):
    bboxes = [woi[1] for woi in wois]
    
    dilate_size = int(size * 0.02)
    dilated_bboxes = list()

    for bbox in bboxes:
        dilated_bboxes.append([
            (bbox[0][0] - dilate_size, bbox[0][1] - dilate_size),
            (bbox[1][0] + dilate_size, bbox[1][1] + dilate_size)
        ])
    
    return dilated_bboxes


if __name__ == '__main__':
    image = cv2.imread(argv[1])
    image = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
    image2 = image[0].copy()
    min_axis = get_min_size_from_image(image)

    wois = get_bbox_and_text_from_raw_image(image)
    dilated_bboxes = clusterize_into_paragraphs(wois, min_axis)

    for bbox in dilated_bboxes:
        image2 = cv2.rectangle(image2, bbox[0], bbox[1], (255, 0, 0), 1)
    
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    cv2.imshow('teste', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
