from dataclasses import dataclass


@dataclass
class TrainingWindow:

    input_ids: list[int]

    labels: list[int]

    loss_mask: list[int]