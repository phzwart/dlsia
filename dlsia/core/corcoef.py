import torch


def cc(a, b):
    """
    Compute a correlation coefficient between two arrays

    :param a: Array a
    :param b: Array b
    :return: A correlation coefficient
    """
    eps = 1e-12
    if a.dtype == torch.float16:
        a = a.type(torch.float32)
        b = b.type(torch.float32)

    ma = torch.mean(a)
    mb = torch.mean(b)
    va = torch.mean(a * a)
    vb = torch.mean(b * b)
    mab = torch.mean(a * b)
    sda = torch.sqrt(va - ma * ma)
    sdb = torch.sqrt(vb - mb * mb)
    result_top = mab - ma * mb
    result_bottom = sda * sdb
    if torch.abs(result_bottom).item() < eps:
        result_bottom = result_bottom * 0 + eps
    return result_top / result_bottom
