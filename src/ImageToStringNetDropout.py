import torch
import torch.nn as nn
import torch.nn.functional as F

classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.:!?'()[]{}<>/\\@#$€£%&~aèéìòù-+°"

class ImageToStringNetDropout(nn.Module):
    def __init__(self):
        super(ImageToStringNetDropout, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Conv1
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            # Conv2
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16 * 4 * 4 + 2, 120),
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, len(classes))
        )

    def forward(self, x, top_margin, bottom_margin):
        x = self.feature_extractor(x)
        
        x = x.view(-1, 16 * 4 * 4)
        x = torch.cat((x, top_margin.view(-1, 1), bottom_margin.view(-1, 1)), dim=1)
        
        x = self.classifier(x)
        return x