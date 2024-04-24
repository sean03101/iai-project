from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits



class Cognex(ImageList):
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d9bca681c71249f19da2/?dl=1"),
        ("amazon", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/edc8d1bba1c740dc821c/?dl=1"),
        ("dslr", "dslr.tgz", "https://cloud.tsinghua.edu.cn/f/ca6df562b7e64850ad7f/?dl=1"),
        ("webcam", "webcam.tgz", "https://cloud.tsinghua.edu.cn/f/82b24ed2e08f4a3c8888/?dl=1"),
    ]
    image_list = {
        "Custom_repeat": "image_list/custom_repeat.txt",
        "Custom_cameraz": "image_list/custom_cameraz.txt",
        "Custom_light": "image_list/custom_lightness.txt",
        "Custom_bright": "image_list/custom_brightness.txt",
        "Custom_blur": "image_list/custom_blur.txt",
        
        "Repeat" : "image_list/repeat.txt",
        "Brightness" : "image_list/brightness.txt",
        "Cameraz" : "image_list/cameraz.txt",
        "Lcondition" : "image_list/lcondition.txt"  
    }
    CLASSES = ['ok', 'scratch', 'fm', 'pin', 'dent', 'glue']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Cognex, self).__init__(root, Cognex.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())