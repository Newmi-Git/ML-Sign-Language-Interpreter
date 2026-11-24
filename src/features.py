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



def hand_vec(hand):
    return extract_features(hand) if hand else [0.0] * 63


def body_vec(pose):
    if not pose: return [0.0] * 12
    mx = (pose[11].x + pose[12].x) / 2
    my = (pose[11].y + pose[12].y) / 2
    sw = np.hypot(pose[11].x - pose[12].x, pose[11].y - pose[12].y) or 1
    out = []
    for i in (11, 12, 13, 14, 15, 16):   # shoulders, elbows, wrists
        out += [(pose[i].x - mx) / sw, (pose[i].y - my) / sw]
    return out


def frame_vec(left, right, pose):         # 63 + 63 + 12 = 138
    return hand_vec(left) + hand_vec(right) + body_vec(pose)

def resample(frames, T=30):
    idx = np.linspace(0, len(frames) - 1, T).round().astype(int)
    return np.array([frames[i] for i in idx], dtype=np.float32)