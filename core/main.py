import numpy as np
from ImageToStringClassifier import ImageToStringClassifier

def process_image(image_pil):
    if image_pil is None:
        return None, ""

    image_np = np.array(image_pil)
    classifier = ImageToStringClassifier(image_np)
    bboxed_image = classifier.preprocessor.get_bboxed_image()
    string_output = classifier.get_string()
    return bboxed_image, string_output