import einops
import numpy as np
import torch
from torch import nn


def set_size(conformal_set):
    """
    Compute the size of the conformal set, i.e. how many elements are in the
    set.
    The shape of the conformal_set tensor should be (N C ...), we count the number of elements
    in channel 1.
    The conformal set is an indicator array, i.e. it contains True / False for each class.

    :param conformal_set: The conformal set
    :type conformal_set: torch.Tensor, type bool
    :return: The size of the conformal set
    :rtype: (N, ....), type int
    """
    result = torch.sum(conformal_set, 1)
    return result


def has_label_in_set(conformal_set, label):
    """
    Checks if a conformal set contains a specific label

    :param conformal_set: The conformal set
    :type conformal_set: torch.Tensor, type bool (N,C, ...)
    :param label: the label of interest
    :type label: torch.Tensor, type int
    :return: does it have this label
    :rtype: torch.Tensor, type bool (N, ...)
    """
    result = conformal_set[:, label, ...]
    return result


def does_not_have_label_in_set(conformal_set, label):
    """
    Checks if a conformal set does not contain a specific label

    :param conformal_set: The conformal set
    :type conformal_set: torch.Tensor, type bool (N,C, ...)
    :param label: the label of interest
    :type label: torch.Tensor, type int
    :return: does it have this label
    :rtype: torch.Tensor, type bool (N, ...)
    """
    result = ~conformal_set
    result = result[:, label, ...]
    return result


conformal_set_metrics = {'size': set_size, 'has': has_label_in_set, 'has_not': does_not_have_label_in_set}


def index_it(scores, labs):
    """
    index the scores

    :param scores: scores
    :type scores:
    :param labs: labels
    :type labs:
    :return: indexed scores
    :rtype:
    """
    scores = scores[range(scores.shape[0]), labs]
    return scores


def _compute_scores(ps, labs):
    """
    Compute scores needed for conformalizing the predictions.
    We only use this to set up the object.

    :param ps: input probabilities
    :type ps: floats
    :param labs: True labels
    :type labs: integers
    :return: scores need to conformalize stuff
    :rtype: floats
    """
    scores = index_it(ps, labs)
    scores = 1 - scores
    return scores.flatten()


class conformalize_classification(nn.Module):
    """
    Conformalize a set of predictions.
    This is performed on hold-out data.
    """

    def __init__(self,
                 alpha,
                 estimated_label_probabilities,
                 true_labels
                 ):
        """
        Build a conformalizing object.

        :param alpha: The power of the test / level of confidence interval
        :type alpha: float
        :param estimated_label_probabilities: estimated probabilities from softmax
        :type estimated_label_probabilities: float
        :param true_labels: The true labels
        :type true_labels: torch.Tensor
        """
        super(conformalize_classification, self).__init__()
        self.alpha = alpha
        self.scores = _compute_scores(estimated_label_probabilities, true_labels)
        self.qhat = self.recalibrate()
        self.labels = torch.arange(0, estimated_label_probabilities.shape[1])

    def recalibrate(self, alpha=None):
        """
        Reset thresholds such that a new confidence limit is returned

        :param alpha: The power of the test / level of confidence interval
        :type alpha: float
        :return: qhat, the score threshold
        :rtype: float
        """
        return_value = True
        if alpha is not None:
            self.alpha = alpha
            return_value = False

        n = self.scores.numel()
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        qhat = np.quantile(self.scores.numpy(), q_level, method='higher')
        if return_value:
            return qhat
        self.qhat = qhat

    def forward(self, p_scores):
        conformal_set = p_scores > 1 - self.qhat
        return conformal_set


def build_conformalizer_classify(model,
                                 testloader,
                                 alpha=0.10,
                                 missing_label=-1,
                                 device='cuda:0',
                                 norma=True):
    """
    Given a model, some test data, a threshold alpha, build me a conformalizer object

    :param model: the input neural network
    :type model: a neural network
    :param testloader: pytorch data loader
    :type testloader: pytorch data loader
    :param alpha: the level alpha
    :type alpha: float
    :param missing_label: missing label - data can be annotated sparsely
    :type missing_label: int, typically -1
    :param device: where do we calculate things
    :type device: 'cpu' / 'cuda:0'
    :param norma: does the network return a normalized score? If False, run an additonal softmax
    :type norma: bool
    :return: a conformalize_classification object
    :rtype: conformalize_classification
    """
    with torch.no_grad():
        labels_all = []
        plabels = []
        for data in testloader:
            inp, lab = data

            pred_p_lab = model.to(device)(inp.to(device)).cpu()
            pred_p_lab = einops.rearrange(pred_p_lab, "N C Y X -> (N Y X) C")
            lab = einops.rearrange(lab, "N C Y X -> (N Y X) C")
            sel = lab != missing_label
            these_labs = lab[sel]
            try:
                these_ps = pred_p_lab[sel]
            except:
                these_ps = pred_p_lab[sel[:, 0], :]

            labels_all.append(these_labs)
            plabels.append(these_ps)
        plabels = torch.concat(plabels)
        if norma:
            plabels = nn.Softmax(1)(plabels)
        labels_all = torch.concat(labels_all)
        cobj = conformalize_classification(alpha, plabels, labels_all)
        return cobj
