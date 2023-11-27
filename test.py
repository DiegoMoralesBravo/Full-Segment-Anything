import os
import numpy as np
from PIL import Image
from mask_generator import SamMaskGenerator
import matplotlib.pyplot as plt
from utils.utils import show_masks
from build_sam import sam_model_registry

def testChido():
    print(os.listdir('/var/data'))


    # auto_to_mask = SamMaskGenerator(sam, stability_score_thresh=0.8)

    # # image upload
    # img = np.array(Image.open("figure/paris2.jpg"))
    # masks = auto_to_mask.generate(img)
    return os.listdir('/var/data')