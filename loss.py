import torch
import torch.nn as nn
import torch.nn.functional as F


class AgeGenderLoss(nn.Module):

    
    def __init__(self,age_weight, gender_weight):
        super(AgeGenderLoss, self).__init__()
        self.age_weight = age_weight
        self.gender_weight = gender_weight

    def forward(self, output, targets):
        gender_output, age_output, = output[:, 0], output[:, 1]
        gender_target, age_target = targets

        gender_loss = torch.mean(-1.0 * (
            torch.log(gender_output) * gender_target +
            torch.log(1.0 - gender_output) * (1.0 - gender_target)
        ))
        
        age_loss = torch.mean((age_output-age_target)**2)
        return self.age_weight * age_loss, self.gender_weight * gender_loss
