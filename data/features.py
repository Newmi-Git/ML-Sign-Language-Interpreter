import numpy as np


def extract_features(hand):
    points = np.array([[p.x, p.y, p.z] for p in hand])
    points -= points[0]
    scale = np.linalg.norm(points[9]) or 1
    return (points / scale).flatten().tolist()
    '''
    mt_list = []
    for point in hand:
        mt_list.extend([point.x,point.y,point.z])
    return mt_list
    '''