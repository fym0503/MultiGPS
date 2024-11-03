import torch
import torch.nn.functional as F 
import torch.nn as nn

class HurdleLoss(nn.BCEWithLogitsLoss):
    '''
    Hurdle loss that incorporates ZCELoss for each output, as well as MSE for
    each output that surpasses the threshold value. This can be understood as
    the negative log-likelihood of a hurdle distribution.

    Args:
      lam: weight for the ZCELoss term (the hurdle).
      thresh: threshold that an output must surpass to be considered turned on.
    '''
    def __init__(self, lam=10.0, thresh=0):
        super().__init__()
        self.lam = lam
        self.thresh = thresh

    def forward(self, pred, target):
        # Verify prediction shape.
        if pred.shape[1] != 2 * target.shape[1]:
            raise ValueError(
                'Predictions have incorrect shape! For HurdleLoss, the'
                ' predictions must have twice the dimensionality of targets'
                ' ({})'.format(target.shape[1] * 2))

        # Reshape predictions, get distributional.
        pred = pred.reshape(*pred.shape[:-1], -1, 2)
        pred = pred.permute(-1, *torch.arange(len(pred.shape))[:-1])
        mu = pred[0]
        p_logit = pred[1]

        # Calculate loss.
        zero_target = (target <= self.thresh).float().detach()
                
        return self.lam * super().forward(p_logit, zero_target) + torch.mean((1 - zero_target) * (target - mu) ** 2)
    

def get_loss(args):
    loss_fn = {}
    loss_fn['cls'] = torch.nn.CrossEntropyLoss()
    
    if args.hurdle:
        loss_fn['recon'] = HurdleLoss(5)
    else:
        loss_fn['recon'] = F.mse_loss
    loss_fn['coo'] = F.mse_loss
        
    return loss_fn