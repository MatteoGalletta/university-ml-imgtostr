import cv2
import sys
import torch 
from torch import Tensor
from ImageToStringPreprocessing import ImageToStringPreprocessing
sys.path.append('../src')
from ImageToStringNet import ImageToStringNet, classes as ImageToStringClasses

class ImageToStringClassifier:
    __MODEL_PATH = '../src/model_weights.pth'
    def __init__(self, image_uploaded):

        self.net = ImageToStringNet()

        # test 
        state_dict = torch.load(self.__MODEL_PATH, map_location=torch.device('cpu'))
        self.net.load_state_dict(state_dict)
        self.net.eval()

        # originale
        #self.net.load_state_dict(torch.load(self.__MODEL_PATH, weights_only=True))
        #self.net.eval()

        #image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_uploaded, cv2.COLOR_BGR2RGB)

        self.preprocessor = ImageToStringPreprocessing(image_rgb)
        # self.postprocessor = ImageToStringPostprocessing()

    def _classify(self):

        device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

        images = Tensor([x['img'][None, ...] for x in self.preprocessor.get_info()]).to(device)
        top_margins = Tensor([x['dist_top'] for x in self.preprocessor.get_info()]).to(device)
        bottom_margins = Tensor([x['dist_bottom'] for x in self.preprocessor.get_info()]).to(device)

        self.net.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = self.net(images, top_margins, bottom_margins)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
        
        return [ImageToStringClasses[i] for i in predicted]
    
    def _add_spaces(self, labels):
        # Add spaces using heuristics
        dist_dx = [x['dist_dx'] for x in self.preprocessor.get_info() if x['dist_dx'] is not None]

        dist_dx_min = min(dist_dx)
        dist_dx_max = max(dist_dx)

        space_indexes = [i for i, v in enumerate(dist_dx) if v is not None and v > (dist_dx_min + dist_dx_max) * 0.6]

        # Insert spaces at the specified positions in the labels list
        labels_with_spaces = labels.copy()

        for idx in reversed(space_indexes):
            labels_with_spaces.insert(idx + 1, ' ')

        return "".join(labels_with_spaces)

    def get_string(self):
        # classify using the model
        labels = list(self._classify())

        # add spaces using heuristics
        labels_with_spaces = self._add_spaces(labels)

        return "".join(labels_with_spaces)
        