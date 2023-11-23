import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

count=0

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha

    def one_hot(self, index, classes):
        size = index.size() + (classes,)
        view = index.size() + (1,)

        mask = torch.Tensor(*size).fill_(0).to(index.device)

        index = index.view(*view)
        ones = 1.

        if isinstance(index, Variable):
            ones = Variable(torch.Tensor(index.size()).fill_(1).to(index.device))
            mask = Variable(mask, volatile=index.volatile)

        return mask.scatter_(1, index, ones)

    def forward(self, input, target):
        y = self.one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        alpha = torch.ones(input.shape[0],input.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * y).sum(dim=1).view(-1,1)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = alpha * loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()

def split_voxels_sparse(indices_ori, features_ori, mask_multi=True, topk=True, threshold=0.5):
    """
        Generate and split the voxels into foreground and background sparse features, based on the predicted importance values.
        Args:
            indices_ori: foreground features indices
            features_ori: foreground features
            mask_multi: bool, whether to multiply the predicted mask to features
            topk: bool, whether to use topk or threshold for selection
            threshold: float, threshold value
    """
    
    mask_voxel = features_ori[:, 0].sigmoid().squeeze(-1)

    if mask_multi:
        features_ori = features_ori * mask_voxel.unsqueeze(-1)

    if topk:
        _, indices = mask_voxel.sort(descending=True)
        indices_fore = indices[:int(mask_voxel.shape[0]*threshold)]
        indices_back = indices[int(mask_voxel.shape[0]*threshold):]
    else:
        indices_fore = mask_voxel > threshold
        indices_back = mask_voxel <= threshold
        

    features_fore = features_ori[indices_fore, 1:]
    coords_fore = indices_ori[indices_fore]

    features_back = features_ori[indices_back, 1:]
    coords_back = indices_ori[indices_back]

    return features_fore, coords_fore, features_back, coords_back, mask_voxel