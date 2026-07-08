import faiss
import numpy as np
import torch


class FAISSDetector:

    def __init__(self,
             index_path="outputs/faiss.index",
             labels_path="outputs/faiss_labels.npy",
             threshold=50.0):

        self.index = faiss.read_index(index_path)
        self.labels = np.load(labels_path)
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
    
    def adversarial_ratio(
            self,
            feature,
            predicted_class,
            k=20):

        distances, indices = self.search(feature, k)

        neighbor_labels = self.labels[indices]

        same_class = np.sum(
            neighbor_labels == predicted_class
        )

        ratio = same_class / k

        unique_classes = len(np.unique(neighbor_labels))

        return {
            "ratio": float(ratio),  #Ratio: how many neighbors match the predicted class.
            "unique_classes": int(unique_classes),  #Unique classes: how diverse the neighborhood is.
            "neighbor_labels": neighbor_labels.tolist(),
            "indices": indices.tolist(),
            "neighbor_distances": distances.tolist()
}

    def detect(self, feature):

        distances, indices = self.search(feature)

        min_distance = distances[0]

        is_ood = min_distance > self.threshold

        return {
            "faiss_distance": float(min_distance),
            "neighbors": indices.tolist(),
            "is_ood": is_ood
        }