import numpy as np
from PIL import Image


def _load_image_rek(path):
	im = Image.open(path)
	return np.array(im)