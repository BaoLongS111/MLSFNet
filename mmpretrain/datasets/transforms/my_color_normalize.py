from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS
import staintools
import cv2
import numpy as np

@TRANSFORMS.register_module()
class MyColorNormalize(BaseTransform):
    def __init__(self, target_img_path):
        self.target_img_path = target_img_path
        self.normalizer = None  # 不在init直接fit

    def _init_normalizer(self):
        if self.normalizer is None:
            target_img = cv2.imread(self.target_img_path)
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            self.normalizer = staintools.StainNormalizer(method='vahadane')
            self.normalizer.fit(target_img)

    def transform(self, results):
        self._init_normalizer()  # 每个worker都能单独初始化
        img = results['img']
        if isinstance(img, np.ndarray) and img.dtype == np.uint8:
            normed_img = self.normalizer.transform(img)
            results['img'] = normed_img
        return results
