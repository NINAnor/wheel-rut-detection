# 4_train

The model used is
torchvision.models.segmentation.deeplabv3_resnet101(). The loss
function is torch.nn.functional.cross_entropy(), with weights found by
the tile creation script.
