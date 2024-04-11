import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Modules containing popular loss functions suitable for image 
segmentation. These loss functions provide varying metrics for judging 
overall network error and performance, a few of which are simple 
averages of others. Additionally, these loss functions leverage torch.nn  
backend functions and are compatible  with GPU implementation and 
Pytorch autograd gradient calculation.

Though this set of functions is intended for binary image segmentation 
of single classes, it acts as a template for averaging multiple classes.

Overviews of these loss functions are detailed in:
    1) https://arxiv.org/pdf/2005.13449.pdf
    2) https://arxiv.org/pdf/2006.14822.pdf
    
Many loss functions modified from :
    1) https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
"""


class L1LossMasked(nn.Module):
    """
    An implementation of a masked L1Loss. Replicates the PyTorch L1Loss
    loss criterion, this time accepting a binary mask as third input
    indicating which pixels to accumulate gradients for.
    """

    def __init__(self):
        super(L1LossMasked, self).__init__()

    def forward(self, inputs, targets, masks):
        """
        :param inputs: tensor of size (N,∗), where N is the batch size
                       and * indicates any number of additional
                       dimensions
        :type inputs: List[float]
        :param targets: tensor of size (N,*), the same size as the input
        :type targets: List[float]
        :param masks: tensor of size (N,*), the same size as the input;
                     indicates which pixels to ignore in loss evaluation
        :type masks: List[bool]
        """

        l1loss = nn.L1Loss(reduction='none')

        loss = l1loss(inputs, targets)

        # gives \sigma_euclidean over unmasked elements
        loss = (loss * masks.float()).sum()

        non_zero_elements = masks.sum()
        l1_loss_val = loss / non_zero_elements

        return l1_loss_val


class MSELossMasked(nn.Module):
    """
    An implementation of a masked mean square error Loss. Replicates the
    PyTorch MSELoss loss criterion, this time accepting a binary mask as
    third input indicating which pixels to accumulate gradients for.
    """

    def __init__(self):
        super(MSELossMasked, self).__init__()

    def forward(self, inputs, targets, masks):
        """
        :param inputs: tensor of size (N,∗), where N is the batch size
                       and * indicates any number of additional
                       dimensions
        :type inputs: List[float]
        :param targets: tensor of size (N,*), the same size as the input
        :type targets: List[float]
        :param masks: tensor of size (N,*), the same size as the input;
                     indicates which pixels to ignore in loss evaluation
        :type masks: List[bool]
        """

        mseloss = nn.MSELoss(reduction='none')

        loss = mseloss(inputs, targets)

        # gives \sigma_euclidean over unmasked elements
        loss = (loss * masks.float()).sum()

        non_zero_elements = masks.sum()
        mse_loss_val = loss / non_zero_elements

        return mse_loss_val


class DiceLoss(nn.Module):
    """
    Creates a criterion that outputs the popular Dice score coefficient
    (DSC), a measure ofboverlap between image regions. It is widely used
    in computer vision for edge detection and is generally compared
    against the ubiquitous Binary Cross-Entropy in segmentation tasks.

    Overall, the Dice loss is a general measure for assessing
    segmentation performance when a ground truth is available, though it
    lacks proper re-weighting when dealing with an imbalance of classes
    (i.e. a single class or two are disproportionately represented).
    """

    # Constructor defines hyperparameters
    def __init__(self, smooth=1):
        """:param float smooth: used to avoid division by 0"""

        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        :param inputs: tensor of size (N,∗), where N is the batch size
                       and * indicates any number of additional
                       dimensions
        :type inputs: List[float]
        :param targets: tensor of size (N,*), the same size as the input
        :type targets: List[float]
        """

        # Comment out if your model contains a sigmoid or equivalent activation
        # inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
               (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Creates a criterion that outputs a combination of Dice score
    coefficient (DSC), a measure of overlap between two sets, and
    binary cross-entropy (BCE), the ubiquitous image segmentation
    measuring dissimilarity between two distributions. This metric's
    use case includes data with lightly imbalanced class epresentation
    (moderate to low foreground-to-background ratio), as it leverages
    the BCE smoothing to enhance the already-flexible performance of
    DSC.
    """

    # Constructor defines hyperparameters
    def __init__(self, smooth=1, alpha=.5):
        """
        :param float smooth: used to avoid division by 0 in Dice loss
                             score
        :param float alpha: weighted distribution of BinaryCrossEntropy
                            loss compared to DiceLoss
        """

        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        :param inputs: tensor of size (N,∗), where N is the batch size
                       and * indicates any number of additional
                       dimensions
        :type inputs: List[float]
        :param targets: tensor of size (N,*), the same size as the input
        :type targets: List[float]
        """
        # Comment out if your model contains a sigmoid or equivalent activation
        # inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum()
                                                             + targets.sum()
                                                             + self.smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce = (self.alpha * bce) + ((1 - self.alpha) * dice_loss)

        return dice_bce


class FocalLoss(nn.Module):
    """
    Creates a criterion that outputs the Focal loss metric, a variation
    of binary cross-entropy. Developed by Facebook AI Research in 2017
    as a means of applying segmentation on examples with low
    foreground-to-background ratios, the FocalLoss criterion
    down-weights the easy-to-detect class decisions and instead focuses
    training on hard negatives.
    """

    # Constructor defines hyperparameters
    def __init__(self, alpha=0.8, gamma=.5):
        """
        :param float alpha: balancing factor typically in range [0,1]
        :param float gamma: focusing parameter of modulating factor
                            (1-exp(-BCE)); FocalLoss is equal to cross
                            entropy when gamma is equal to 1
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        :param inputs: tensor of size (N,∗), where N is the batch size
                       and * indicates any number of additional
                       dimensions
        :type inputs: List[float]
        :param targets: tensor of size (N,*), the same size as the input
        :type targets: List[float]
        """
        # Comment out if your model contains a sigmoid or equivalent activation
        # inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce

        return focal_loss


class TverskyLoss(nn.Module):
    """
    Creates a criterion that outputs the Tversky index loss, a
    generalization of the Dice loss coefficient, a measure of overlap
    between two sets that is widely popular in computer vision. The
    Tversky loss index aims to achieve a better precision-to-recall
    trade-off by adjusting how harshly false positives (fp) and false
    negatives (fn) are penalized.

    Hyperparameters alpha and beta control how harshly fps and fns are
    penalized, respectively, allowing the user to place higher emphasis
    on precision/recal with a larger alpha/beta value. For reference,
    setting alpha=beta=0.5 results in the DiceLoss criterion.
    """

    # Constructor defines hyperparameters
    def __init__(self, smooth=1, alpha=0.75, beta=0.25):
        """
        :param float smooth: used to avoid division by 0"
        :param float alpha: parameter that weights penalty of detecting
                            a false positive
        :param float beta: parameter that weights penalty of detecting a
                           false negative
        """
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        """
        :param inputs: tensor of size (N,∗), where N is the batch size
                       and * indicates any number of additional
                       dimensions
        :type inputs: List[float]
        :param targets: tensor of size (N,*), the same size as the input
        :type targets: List[float]
        """
        # Comment out if your model contains a sigmoid or equivalent activation
        # inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True positives, false positives, and false negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()

        tversky_loss = (tp + self.smooth) / (tp + self.alpha
                                             * fp + self.beta
                                             * fn + self.smooth)

        return 1 - tversky_loss


class FocalTverskyLoss(nn.Module):
    """
    Creates a criterion that combines the customizable
    precision-to-recall penalty decisions of the Tversky loss (TL) index
    with the Focal loss (FL) advantages of dealing with hard examples
    with low foreground-to-background ratios.

    The FocalTverskyLoss (FTL) is mainly comprised of TL loss, a
    generalization of the Dice loss coefficient, a measure of overlap
    between two sets that is widely popular in computer vision. The TL
    index aims to achieve a better precision-to-recall trade-off by
    adjusting how harshly false positives (fp) and false negatives (fn)
    are penalized. Hyperparameters alpha and  beta control how harshly
    fps and fns are penalized, respectively, allowing the user to place
    higher emphasis on precision/recal with a larger alpha/beta value.
    For reference, setting alpha=beta=0.5 results in the DiceLoss
    criterion.

    Once the TL index is computed, hyperparameter gamma is introduced
    which down-weights the easy-to-detect class decisions and instead
    focuses training on hard negatives.
    """

    # Constructor defines hyperparameters
    def __init__(self, smooth=1, alpha=0.5, beta=0.5, gamma=.8):
        """
        :param float smooth: used to avoid division by 0
        :param float alpha: parameter that weights penalty of detecting
                            a false positive
        :param float beta: parameter that weights penalty of detecting a
                           false negative
        :param float gamma: focusing parameter of modulating factor
                            (1 - TL))
        """

        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        :param inputs: tensor of size (N,∗), where N is the batch size
                       and * indicates any number of additional
                       dimensions
        :type inputs: List[float]
        :param targets: tensor of size (N,*), the same size as the input
        :type targets: List[float]
        """
        # Comment out if your model contains a sigmoid or equivalent activation
        # inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True positives, false positives, and false negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()

        tversky_loss = (tp + self.smooth) / (tp + self.alpha * fp +
                                             self.beta * fn + self.smooth)
        focal_tversky_loss = (1 - tversky_loss) ** self.gamma

        return focal_tversky_loss


class CombinedLossWithTVNorm(nn.Module):
    """
    A custom loss function combining a base loss and total variation (TV) norm regularization.

    Args:
        base_loss_fn (torch.nn.Module): The base loss function for the primary task.
        tv_weights (torch.Tensor): Weights to control the impact of the TV regularization on each channel.

    Attributes:
        tv_weights (torch.Tensor): Weights to control the impact of the TV regularization on each channel.
        base_loss_fn (torch.nn.Module): The base loss function for the primary task.

    Methods:
        tv_norm(input_tensor):
            Calculates the total variation norm regularization for the input tensor.

        forward(input, target):
            Computes the combined loss with the base loss and TV norm regularization.

    Example:
        base_loss_fn = torch.nn.MSELoss()
        tv_weights = torch.tensor([0.1, 0.2, 0.3])  # Example TV weights for 3 channels
        combined_loss = CombinedLossWithTVNorm(base_loss_fn, tv_weights)
    """

    def __init__(self, base_loss_fn, tv_weights):
        super(CombinedLossWithTVNorm, self).__init__()
        self.tv_weights = tv_weights
        self.base_loss_fn = base_loss_fn

    def tv_norm(self, input_tensor):
        """
        Calculate the total variation norm regularization for the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor to compute the TV norm for.

        Returns:
            torch.Tensor: The total variation norm regularization.
        """
        batch_size, channels, y, x = input_tensor.shape
        diff_i = torch.abs(input_tensor[:, :, 1:, :] - input_tensor[:, :, :-1, :])
        diff_j = torch.abs(input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :-1])

        # Applying the weights for each channel
        diff_i = (diff_i * self.tv_weights.view(1, channels, 1, 1)).sum()
        diff_j = (diff_j * self.tv_weights.view(1, channels, 1, 1)).sum()

        return (diff_i + diff_j) / (batch_size * y * x)

    def forward(self, input, target):
        """
        Compute the combined loss with the base loss and TV norm regularization.

        Args:
            input (torch.Tensor): The model's predicted output.
            target (torch.Tensor): The ground truth target.

        Returns:
            torch.Tensor: The combined loss with base loss and TV norm regularization.
        """
        base_loss = self.base_loss_fn(input, target)
        tv_loss = self.tv_norm(input)
        return base_loss + tv_loss

class CombinedLossWithTVNorm3D(nn.Module):
    """
    A custom loss function combining a base loss and total variation (TV) norm regularization.

    Args:
        base_loss_fn (torch.nn.Module): The base loss function for the primary task.
        tv_weights (torch.Tensor): Weights to control the impact of the TV regularization on each channel.

    Attributes:
        tv_weights (torch.Tensor): Weights to control the impact of the TV regularization on each channel.
        base_loss_fn (torch.nn.Module): The base loss function for the primary task.

    Methods:
        tv_norm(input_tensor):
            Calculates the total variation norm regularization for the input tensor.

        forward(input, target):
            Computes the combined loss with the base loss and TV norm regularization.

    Example:
        base_loss_fn = torch.nn.MSELoss()
        tv_weights = torch.tensor([0.1, 0.2, 0.3])  # Example TV weights for 3 channels
        combined_loss = CombinedLossWithTVNorm(base_loss_fn, tv_weights)
    """

    def __init__(self, base_loss_fn, tv_weights):
        super(CombinedLossWithTVNorm3D, self).__init__()
        self.tv_weights = tv_weights
        self.base_loss_fn = base_loss_fn

    def tv_norm(self, input_tensor):
        """
        Calculate the total variation norm regularization for the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor to compute the TV norm for.

        Returns:
            torch.Tensor: The total variation norm regularization.
        """
        batch_size, channels, z, y, x = input_tensor.shape
        sm = nn.Softmax(dim=1)
        sm_input_tensor = sm(input_tensor)
        diff_i = torch.abs(sm_input_tensor[:, :, 1:, :, :] - sm_input_tensor[:, :, :-1, :, :])
        diff_j = torch.abs(sm_input_tensor[:, :, :, 1:, :] - sm_input_tensor[:, :, :, :-1, :])
        diff_k = torch.abs(sm_input_tensor[:, :, :, :, 1:] - sm_input_tensor[:, :, :, :, :-1])

        # Applying the weights for each channel
        diff_i = (diff_i * self.tv_weights.view(1, channels, 1, 1, 1)).sum()
        diff_j = (diff_j * self.tv_weights.view(1, channels, 1, 1, 1)).sum()
        diff_k = (diff_k * self.tv_weights.view(1, channels, 1, 1, 1)).sum()

        return (diff_i + diff_j + diff_k) / (batch_size * z * y * x)

    def forward(self, input, target):
        """
        Compute the combined loss with the base loss and TV norm regularization.

        Args:
            input (torch.Tensor): The model's predicted output.
            target (torch.Tensor): The ground truth target.

        Returns:
            torch.Tensor: The combined loss with base loss and TV norm regularization.
        """
        base_loss = self.base_loss_fn(input, target)
        tv_loss = self.tv_norm(input)
        return base_loss + tv_loss

def tst():
    """
    Defines and test several Mixed Scale Dense Networks consisting of 2D
    convolutions, provides a printout of the network, and checks to make
    sure tensors pass through the network
    """

    lossD = DiceLoss()
    lossDBCE = DiceBCELoss()
    lossF = FocalLoss()
    lossT = TverskyLoss()
    lossFT = FocalTverskyLoss()
    lossL1Masked = L1LossMasked()
    lossMSEMasked = MSELossMasked()

    m = nn.Sigmoid()
    input_ = torch.randn(2, 4, requires_grad=True)
    target = torch.ones(2, 4)
    mask = torch.randint(2, (2, 4))
    mask[:, 1:] = 0

    mask = mask > 0
    print(m(input_))
    print(mask)

    outputD = lossD(m(input_), target)
    outputDBCE = lossDBCE(m(input_), target)
    outputF = lossF(m(input_), target)
    outputT = lossT(m(input_), target)
    outputFT = lossFT(m(input_), target)
    outputL1Masked = lossL1Masked(m(input_), target, mask)
    outputMSEMasked = lossMSEMasked(m(input_), target, mask)

    # print(outputL1Masked)

    outputD.backward()
    outputDBCE.backward()
    outputF.backward()
    outputT.backward()
    outputFT.backward()
    outputL1Masked.backward()
    outputMSEMasked.backward()


if __name__ == "__main__":
    tst()
