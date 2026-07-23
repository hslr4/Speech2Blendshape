import numpy as np

UPPER_MASK = [
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 49, 50
]

MOUTH_MASK = [
    23, 25, 27, 28, 31, 37, 39, 40, 41, 42, 47, 48
]

train_label_means = np.array([[[0.06682316,0.06681694,0.25371084,0.12956057,0.12956651,0.02150766
 ,0.17319861,0.18742893,0.07452572,0.07430915,0.08153474,0.08158347
 ,0.1344843,0.09138267,0.03961422,0.07395752,0.10274948,0.10300316
 ,0.25183038,0.2521907,0.21227364,0.2121901,0.12837101,0.02263843
 ,0.19499805,0.01192453,0.12104702,0.12321976,0.11982337,0.00546072
 ,0.00777778,0.23678,0.01196589,0.33497216,0.33919418,0.04842478
 ,0.05234776,0.06621926,0.00688575,0.05919219,0.06104041,0.04430697
 ,0.22396134,0.22447881,0.20307746,0.26737597,0.26826902,0.11037954
 ,0.10609268,0.15481554,0.16048624]]])

def mae(pred, gt):
    return np.abs((pred - gt)).mean()

def masked_mae(pred, gt, mask):
    return np.abs((pred[..., mask] - gt[..., mask])).mean()

def mse(pred, gt):
    return np.linalg.norm(pred - gt, axis=(-1)).mean()

def masked_mse(pred, gt, mask):
    return np.linalg.norm(pred[..., mask] - gt[..., mask], axis=(-1)).mean()

def rae(pred, gt):
    means = train_label_means.copy().repeat([gt.shape[1]], axis=1)
    means = means.repeat([gt.shape[0]], axis=0)
    rae = np.abs((pred - gt)) / np.abs((pred - means))
    return rae.mean()
