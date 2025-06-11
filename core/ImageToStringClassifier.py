import cv2
import sys
import torch 
import numpy as np
from torch import Tensor
from ImageToStringPreprocessing import ImageToStringPreprocessing
from ImageToStringPostprocessing import ImageToStringPostprocessing
sys.path.append('../src')
from ImageToStringNet import ImageToStringNet, classes as ImageToStringClasses
from ImageToStringNetDropout import ImageToStringNetDropout

class ImageToStringClassifier:
    # __MODEL_PATH = '../src/model_weights.pth'
    # __MODEL_PATH = '../src/model_weights_v2.pth'
    __MODEL_PATH = '../src/model_weights_v7.pth'

    # __NET = ImageToStringNet()
    __NET = ImageToStringNetDropout()

    def __init__(self, image_uploaded):

        self.net = self.__NET

        device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

        state_dict = torch.load(self.__MODEL_PATH, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

        #image_uploaded = cv2.cvtColor(image_uploaded, cv2.COLOR_BGR2RGB)

        self.preprocessor = ImageToStringPreprocessing(image_uploaded)
        self.postprocessor = ImageToStringPostprocessing()

    def _classify(self):

        device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

        images_np = np.array([x['img'][None, ...] for x in self.preprocessor.get_info()])
        images = Tensor(images_np).to(device)
        top_margins = Tensor([x['dist_top'] for x in self.preprocessor.get_info()]).to(device)
        bottom_margins = Tensor([x['dist_bottom'] for x in self.preprocessor.get_info()]).to(device)

        self.net.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = self.net(images, top_margins, bottom_margins)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
        
        return [ImageToStringClasses[i] for i in predicted]
    
    def get_string(self):
        # classify using the model
        labels = list(self._classify())
        info_w_char = [{**x[0], 'char': x[1], 'index': i} for i, x in enumerate(zip(self.preprocessor.get_info(), labels))]
        info_w_char2 = self.postprocessor.heuristics_adjust(info_w_char)
        char_list = [entry['char'] for entry in info_w_char2]
        labels_with_spaces = self.postprocessor.heuristics_spaces(info_w_char2, char_list)
        return "".join(labels_with_spaces).replace("''", '"')