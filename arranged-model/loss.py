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

        gender_loss = self.gender_weight * F.binary_cross_entropy(
            gender_output,
            gender_target,
            reduction='mean'
        )

        age_loss = self.age_weight * F.smooth_l1_loss(
            age_output,
            age_target,
            reduction='mean'
        )
        return gender_loss, age_loss
