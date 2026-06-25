import faiss
import numpy as np
import torch


class FAISSDetector:

    def __init__(self,
                 index_path="outputs/faiss.index",
                 threshold=50.0):

        self.index = faiss.read_index(index_path)
        self.threshold = threshold

    def search(self, feature, k=5):

        if isinstance(feature, torch.Tensor):
            feature = feature.cpu().numpy()

        feature = feature.astype(np.float32)

        if len(feature.shape) == 1:
            feature = feature.reshape(1, -1)

        distances, indices = self.index.search(
            feature,
            k
        )

        return distances[0], indices[0]

    def detect(self, feature):

        distances, indices = self.search(feature)

        min_distance = distances[0]

        is_ood = min_distance > self.threshold

        return {
            "distance": float(min_distance),
            "neighbors": indices.tolist(),
            "is_ood": is_ood
        }