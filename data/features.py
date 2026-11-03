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

def frame_features(hand):
    return extract_features(hand), hand[0].x, hand[0].y

def window_to_arr(window):
    wx0, wy0 = window[0][1], window[0][2]
    rows = [f + [wx-wx0, wy-wy0] for f, wx, wy in window]
    return np.array(rows, dtype=np.float32)


