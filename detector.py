import json
import os
import re
from dataclasses import dataclass, field

import cv2
import imutils
import keras_ocr as koc
import numpy as np
import pandas as pd
import requests


@dataclass
class Detector:
    image_path: str = 'images_to_process'
    url_json_path: str = None
    target_brands_path: str = None

    def __post_init__(self):
        self._validate()
        self.url_mapper = {}
        if self.image_path not in os.listdir():
            os.mkdir(self.image_path)
        self.pipe = koc.pipeline.Pipeline()
        self._download_images()
        self._read_target_brands()

    def _validate(self):
        assert self.url_json_path in os.listdir()
        assert self.target_brands_path in os.listdir()

    def _download_images(self):
        with open(self.url_json_path) as f:
            content = f.read()
        url_dict = json.loads(content)
        url_list = url_dict['url_list']
        for n, url in enumerate(url_list):
            try:
                img_data = requests.get(url).content
                self.url_mapper[f'{n}.jpg'] = url
                with open(f'./{self.image_path}/{n}.jpg', 'wb') as handler:
                    handler.write(img_data)
            except:
                pass

    def _read_target_brands(self):
        with open(self.target_brands_path) as f:
            content = f.read()
        brand_dict = json.loads(content)
        self.brand_list = brand_dict['brand_list']

    def _read_images_to_matrices(self):
        img_files = [a for a in os.listdir(
            self.image_path) if re.search('jpg', a) is not None]
        images = [
            koc.tools.read(os.path.join(self.image_path, img)) for img in img_files
        ]
        file_to_img = {img_files[n]: [images[n]]
                       for n in range(len(img_files))}
        return file_to_img

    def _rotate_img(self, img, rotation):
        return [imutils.rotate_bound(img[0], angle=rotation)]

    def _generate_pred_dict(self, img_dict, rotations):
        prediction_groups = {}
        if rotations is None:
            for img_name, img in img_dict.items():
                preds = self.pipe.recognize(img)[0]
                prediction_groups[img_name] = [pred[0] for pred in preds]
        else:
            for img_name, img in img_dict.items():
                preds = self.pipe.recognize(img)[0]
                prediction_groups[img_name] = [pred[0] for pred in preds]
                for duo in preds:
                    box = duo[1]
                    w, h = (round(box[:, 0].max() - box[:, 0].min()),
                            round(box[:, 1].max() - box[:, 1].min()))
                    if h > w:
                        for rot in rotations:
                            rotated_img = self._rotate_img(img, rotation=rot)
                            new_preds = self.pipe.recognize(rotated_img)[0]
                            prediction_groups[img_name] += [pred[0]
                                                            for pred in new_preds]

        return {key: list(np.unique(value)) for key, value in prediction_groups.items()}

    def _image_to_brand(self, image_to_pred, brands):
        brands_pat = ('|').join(brands)
        image_to_brands = {}
        for key, words in image_to_pred.items():
            words_joined = ' '.join(words)
            found_brands = re.findall(brands_pat, words_joined)
            image_to_brands[key] = found_brands
        return image_to_brands

    def transform(self, rotations=[-45, -90, 45, 90], url_as_keys=True, addtional_brands=[]):
        brand_list = self.brand_list + addtional_brands
        file_to_img = self._read_images_to_matrices()
        pred_dict = self._generate_pred_dict(file_to_img, rotations=rotations)
        image_to_brands = self._image_to_brand(pred_dict, brand_list)
        if url_as_keys:
            image_to_brands = {
                self.url_mapper[key]: image_to_brands[key] for key in image_to_brands}
        return image_to_brands


if __name__ == '__main__':
    detector = Detector(image_path='test_images', url_json_path='example_urls.json',
                        target_brands_path='example_target_brands.json')
    return_dict = detector.transform()
    print('ok')
