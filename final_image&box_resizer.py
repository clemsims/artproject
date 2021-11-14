## Objectif : parcourir les 260 images et leur .json (les box tracées en orignial size) afin de resize en 224x224 les images ET leur box !! 

## En pratique : Use CHITRA. It can rescale your bounding box automatically based on the new image size.(chitra uses imgaug internally)
https://stackoverflow.com/questions/18805348/how-rename-the-images-in-folder/18805442

def normalizer(img_name): 

import chitra 
import matplotlib.pyplot as plt

image = chitra(img_path, box, label)

image.resize_image_with_bbox((224, 224))

print('rescaled bbox:', image.bounding_boxes)
plt.imshow(image.draw_boxes())