from pathlib import Path
import json

import matplotlib.pyplot as plt


class TrainingVisualizer:
    """
    Creates training plots from the JSON log.
    """

    def __init__(
        self,
        log_file: str,
        output_dir: str = "plots",
    ):

        self.log_file = Path(log_file)

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history = self._load_history()

    def _load_history(
        self,
    ) -> list[dict]:

        with open(
            self.log_file,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)
        
    def _plot(
        self,
        x,
        y,
        title,
        ylabel,
        filename,
    ):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            x,
            y,
            marker="o",
        )

        plt.title(title)

        plt.xlabel("Epoch")

        plt.ylabel(ylabel)

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.output_dir / filename
        )

        plt.close()

    def plot_loss(self):

        epochs = [
            row["epoch"]
            for row in self.history
        ]

        train_loss = [
            row["train_loss"]
            for row in self.history
        ]

        validation_loss = [
            row["validation_loss"]
            for row in self.history
        ]

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            epochs,
            train_loss,
            marker="o",
            label="Train",
        )

        plt.plot(
            epochs,
            validation_loss,
            marker="o",
            label="Validation",
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Training Loss")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            "loss_curve.png"
        )

        plt.close()

    def plot_perplexity(self):

        epochs = [
            row["epoch"]
            for row in self.history
        ]

        train = [
            row["train_perplexity"]
            for row in self.history
        ]

        validation = [
            row["validation_perplexity"]
            for row in self.history
        ]

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            epochs,
            train,
            marker="o",
            label="Train",
        )

        plt.plot(
            epochs,
            validation,
            marker="o",
            label="Validation",
        )

        plt.xlabel("Epoch")

        plt.ylabel("Perplexity")

        plt.title("Training Perplexity")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.output_dir /
            "perplexity_curve.png"
        )

        plt.close()

    def plot_learning_rate(self):

        epochs = [
            row["epoch"]
            for row in self.history
        ]

        learning_rate = [
            row["learning_rate"]
            for row in self.history
        ]

        self._plot(
            epochs,
            learning_rate,
            "Learning Rate",
            "Learning Rate",
            "learning_rate_curve.png",
        )

    def generate(self):

        self.plot_loss()

        self.plot_perplexity()

        self.plot_learning_rate()

        print()

        print(
            "✓ Training plots saved to",
            self.output_dir,
        )