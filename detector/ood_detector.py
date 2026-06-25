from faiss_module.faiss_index import FAISSDetector  #nuha

import torch

from detector.mahalanobis import (
    MahalanobisScorer
)


class OODDetector:

    def faiss_detect(self, feature): #nua
        return self.faiss.detect(feature) #nua

    def __init__(self):

        self.scorer = MahalanobisScorer(
            "outputs/class_means.pt",
            "outputs/covariance.pt"
        )

        self.threshold = torch.load(
            "outputs/threshold.pt"
        )
        
        #nua
        self.faiss = FAISSDetector(
            "outputs/faiss.index",
            threshold=50
        )
        #nua

    def detect(self, feature):

        class_id, distance = (
            self.scorer.predict(
                feature
            )
        )

        is_ood = (
            distance >
            self.threshold.item()
        )

        return {
            "class_id": class_id,
            "distance": distance,
            "is_ood": is_ood
        }
    