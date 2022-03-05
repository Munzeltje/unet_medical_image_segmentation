import torch
import numpy as np
from PIL import Image
import os
from sklearn.metrics import classification_report
#np.set_printoptions(threshold=np.nan)

class Metric(object):
    """
    validation metrics.
    """
    def __init__(self, compute_jaccard=False, device="cuda"):
        self.device = device

    def multi_dice(self, input, target, n_labels):
        # Compute the dice per class
        # and take the mean over all the classes save for the background dice
        dices = []
        for i in range(n_labels):
            dice = self.dice((input==i).float(), (target==i).float())
            dices.append(dice.item())
        print(dices)
        return np.mean(dices[1:])

    def dice(self, input, target):
        '''
        Given an input and target compute the dice score
        between both. Dice: 1 - (2 * |A U B| / (|A| + |B|))
        '''
        eps = 1e-6
        if len(input.shape) > 1:
            input, target = input.view(-1), target.view(-1)

        inter = torch.dot(input, target)
        union = torch.sum(input) + torch.sum(target) + eps
        dice = (2 * inter.float() + eps) / union.float()
        return dice

    def pixel_wise(self, input, target):
        """
        Regular pixel_wise accuracy metric, we just
        compare the number of positives and divide it
        by the number of pixels.
        """
        # Flatten the matrices to make it comparable
        input = input.view(-1)
        target = target.view(-1)
        correct = torch.sum(input==target)
        return (correct.item() / len(input))

    def evaluate(self, net, val_loader, binary=False):

        """"
        Given the trained network, and the validation set, compute the dice score.
        """
        print("Initiated Metric Evaluation")

        net.eval()
        preds_real, masks_real = [], []
        with torch.no_grad():
            for i, (img, true_masks) in enumerate(val_loader):
                img = img.to(self.device)#.permute(0,3,1,2)
                true_masks = true_masks.to(self.device)

                B = true_masks.shape[0]
                patch_labels = torch.zeros(B).to(self.device)

                predictions = net(img)
                predictions = (torch.argmax(predictions, dim=1)).float()

                preds_real += list(predictions.view(-1).cpu().numpy())
                masks_real += list((true_masks.view(-1)).float().cpu().numpy())

        preds, masks = torch.tensor(preds_real), torch.tensor(masks_real)
        n_labels = 4
        seg_dice = self.multi_dice(preds, masks, n_labels)
        acc = self.pixel_wise(preds, masks)

        return acc, seg_dice


