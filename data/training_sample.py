from dataclasses import dataclass


@dataclass
class TrainingSample:
    """
    One training example produced from
    a conversation.
    """

    input_ids: list[int]

    labels: list[int]

    loss_mask: list[int]

    conversation_length: int