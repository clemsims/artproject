import pixellib
from pixellib.custom_train import instance_custom_training


train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 1)
train_maskrcnn.load_dataset("Dataset")
train_maskrcnn.evaluate_model("mask_rcnn_models")