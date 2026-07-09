import json
from pathlib import Path


class TrainingLogger:
    """
    Logs training metrics to a JSON file.
    """

    def __init__(
        self,
        log_file: str,
    ):

        self.log_file = Path(
            log_file
        )

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history = []

    def log(
        self,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        train_perplexity: float,
        validation_perplexity: float,
        learning_rate: float,
        is_best_model: bool = False,
        validation_improved: bool = False,
    ):
        """
        Logs one training epoch.
        """

        entry = {

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                validation_loss,

            "train_perplexity":
                train_perplexity,

            "validation_perplexity":
                validation_perplexity,

            "learning_rate":
                learning_rate,

            "is_best_model":
                is_best_model,
            
            "validation_improved":
                validation_improved,
        }

        self.history.append(
            entry
        )

        self.save()

    def save(
        self,
    ):
        """
        Saves all logged metrics.
        """

        with open(
            self.log_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4,
            )