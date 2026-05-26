import numpy as np
import json
import os


def myRot(points, theta):
    pp = np.array(points)
    rot_list = np.zeros(pp.shape)
    rot_list[:, 0] = pp[:, 0] * np.cos(theta) - pp[:, 1] * np.sin(theta)
    rot_list[:, 1] = pp[:, 0] * np.sin(theta) + pp[:, 1] * np.cos(theta)
    return rot_list

def randomRot(points, n, rot=True):
    if rot:
        rot_list = []
        for i in range(n):
            theta = np.random.uniform(0, 2 * np.pi)
            rotated = myRot(points, theta)
            rot_list.append(rotated)
        return rot_list
    else:
        return [np.array(points) for _ in range(n)]
    
def readPointsJson(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data

def saveRotatedVersions(path, chosen_shapes, n_rotations=9):
    points_list = readPointsJson(path)
    dic = {}
    for t in chosen_shapes:
        points = points_list[str(t)]
        rot_versions = randomRot(points, n_rotations, rot=True)
        dic[str(t)] = points
        for i, version in enumerate(rot_versions):
            dic[f"{t}_{i}"] = version.tolist()
    output_dir = os.path.dirname(path)
    output_path = os.path.join(output_dir, f"rotatedSampleByArea5.json")
    with open(output_path, 'w') as f:
        json.dump(dic, f, indent=4)

path = "data/combinedPtsNormalizedByArea.json"
# chose 10 unique numbers randomly from 1 to 1100
chosen_shapes = np.random.choice(range(1, 1101), size=10, replace=False)
# chosen_shapes = [1, 18, 19, 20, 26, 41, 40, 123, 157, 672]
print("Chosen shapes:", chosen_shapes)
saveRotatedVersions(path, chosen_shapes, n_rotations=9)


