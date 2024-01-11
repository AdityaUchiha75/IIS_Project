import torch
from torch import nn
from pathlib import Path
from time import sleep
from furhat.furhatConfig import LOOK_DOWN, LOOK_BACK


class MLP(nn.Module):
    def __init__(self, features_in=2, features_out=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(features_in, 10),
            nn.ReLU(),
            nn.Linear(10, features_out),
        )

    def forward(self, input):
        return self.net(input)
    
def best_emo_model(local_path):
    return torch.load(Path(local_path) )

def return_emo(loc):
    expression = {"anger": 6, "disgust": 5 , "fear": 4, "happiness": 1, "neutral": 0, "sadness": 2, "surprise": 3}
    for key,_ in expression.items():
        if expression[key] == loc:
            return key

# if __name__ == '__main__':
#     print(DIR_PATH)
#     t = best_emo_model('models\\best_model_12.pt')