import csv
from datetime import datetime
from faiss_module.faiss_index import FAISSDetector  #nuha

import torch

from detector.mahalanobis import (
    MahalanobisScorer
)



CIFAR_CLASSES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


class OODDetector:

    def faiss_detect(self, feature, predicted_class):

        faiss_result = self.faiss.detect(feature)

        ratio_result = self.faiss.adversarial_ratio(
            feature,
            predicted_class
        )

        ratio = ratio_result["ratio"]

        if ratio > 0.5:
            sample_type = "Adversarial"
        else:
            sample_type = "OOD"

        return {
            **faiss_result,
            **ratio_result,
            "sample_type": sample_type
        }

    def __init__(self):

        self.scorer = MahalanobisScorer(
            "outputs/class_means.pt",
            "outputs/covariance.pt"
        )

        # Original threshold
        # self.threshold = torch.load("outputs/threshold.pt")

        # Temporary threshold for testing Stage 2
        self.threshold = 550.0
        
        #nua
        self.faiss = FAISSDetector(
            "outputs/faiss.index",
            threshold=50
        )
        #nua

    def log_detection(self, result):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        predicted_class = CIFAR_CLASSES[result["class_id"]]

        mahalanobis_distance = result["distance"]

        threshold = self.threshold

        faiss_distance = result.get("faiss_distance", "")

        adversarial_ratio = result.get("ratio", "")

        final_decision = "ID"

        if result.get("sample_type") == "OOD":
            final_decision = "OOD"

        elif result.get("sample_type") == "Adversarial":
             final_decision = "Adversarial"

        row = [
            timestamp,
            predicted_class,
            mahalanobis_distance,
            threshold,
            faiss_distance,
            adversarial_ratio,
            final_decision
        ]

            # -------------------------------
            # Log every prediction
            # -------------------------------

        with open("logs/detection_log.csv",
                    "a",
                newline="") as file:

            writer = csv.writer(file)
            writer.writerow(row)

            # -------------------------------
            # Log only OOD
            # -------------------------------

        if final_decision == "OOD":

            with open("logs/ood_log.csv",
                        "a",
                    newline="") as file:

                writer = csv.writer(file)
                writer.writerow(row)

            # -------------------------------
            # Log only Adversarial
            # -------------------------------

        elif final_decision == "Adversarial":

            with open("logs/adversarial_log.csv",
                        "a",
                    newline="") as file:

                writer = csv.writer(file)
                writer.writerow(row)

    def detect(self, feature):

        class_id, distance = (
            self.scorer.predict(
                feature
            )
        )

        is_ood = (
            distance >
            self.threshold
        )

        result = {
            "class_id": class_id,
            "distance": distance,
            "is_ood": is_ood
        }

        # Only run Stage 2 if Stage 1 flags the sample
        if is_ood:

            stage2 = self.faiss_detect(
                feature,
                class_id
            )

            result.update(stage2)

        self.log_detection(result)

        return result
    