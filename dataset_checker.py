import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from modules.utils import set_memory_growth
from modules.dataset import load_cifar10_dataset


set_memory_growth()

dataset = load_cifar10_dataset(batch_size=1, split='train', shuffle=False,
                               using_normalize=False)

for (img, labels)in dataset:
    img = img.numpy()[0]
    print(img.shape, labels.shape, labels.numpy())

    cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(0) == ord('q'):
        exit()
