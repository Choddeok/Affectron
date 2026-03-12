import numpy as np
from operator import itemgetter
from sklearn import metrics


def cosine_similarity(x, y):
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return (tunedThreshold, eer, fpr, fnr)

def jaccard_distance(x, y, thr=None, topk=None, use_abs=True):
    """
    x, y: 1D 또는 임의 shape의 배열(자동 flatten)
    thr:  연속 벡터를 이진화할 임계값. None이면 '!= 0' 기준으로 이진화
    topk: 상위 K개 인덱스만 집합으로 선택 (thr보다 우선 적용)
    use_abs: topk 선택 시 절댓값 기준 여부

    return: Jaccard distance (0.0 ~ 1.0)
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    # 1) top-k 집합 선택 (원하면 절댓값 기준)
    if topk is not None:
        if topk <= 0:
            raise ValueError("topk must be positive.")
        score_x = np.abs(x) if use_abs else x
        score_y = np.abs(y) if use_abs else y
        k = min(topk, x.size)
        Ax = set(np.argpartition(score_x, -k)[-k:])
        Ay = set(np.argpartition(score_y, -k)[-k:])
    else:
        # 2) 임계값 기반 이진화 → 활성 인덱스 집합
        if thr is None:
            bx = (x != 0)
            by = (y != 0)
        else:
            bx = (x >= thr)
            by = (y >= thr)
        Ax = set(np.nonzero(bx)[0])
        Ay = set(np.nonzero(by)[0])

    union = Ax | Ay
    inter = Ax & Ay
    if len(union) == 0:
        # 두 벡터가 모두 '비활성'(공집합)이면 완전일치로 간주
        return 0.0
    j_index = len(inter) / len(union)
    return 1.0 - j_index  # Jaccard distance