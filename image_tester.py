import pixellib
from pixellib.instance import custom_segmentation
import os

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 1, class_names= ["BG", "painting"])

def image_test(model,filename):
    segment_image.load_model(model) ## insérer son modèle !!
    x = os.path.splitext(filename)[0] + "_segmented.jpg"
    print(x)
    segment_image.segmentImage(filename, show_bboxes=True, output_image_name=x)

image_test("mask_rcnn_models\mask_rcnn_model.002-2.253000.h5","painting_frames150.jpg")
