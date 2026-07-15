import torch
from torch.utils.data import Dataset

from data.training_sample import TrainingSample
from training.training_window import TrainingWindow
from tokenizer.tokenizer import create_tokenizer


class ConversationDataset(Dataset):
    """
    Dataset for GPT-style language modeling.

    Each conversation is converted into fixed-length
    training windows.

    Windows never cross conversation boundaries.
    """

    def __init__(
        self,
        samples: list[TrainingSample],
        context_length: int,
        stride: int,
    ):

        self.context_length = context_length
        self.stride = stride

        self.windows: list[TrainingWindow] = []

        tokenizer = create_tokenizer()

        self.pad_token_id = tokenizer.pad_token_id

        for sample in samples:

            input_ids = sample.input_ids
            labels = sample.labels
            loss_mask = sample.loss_mask

            #
            # Safety check.
            #

            assert (
                len(input_ids)
                == len(labels)
                == len(loss_mask)
            )

            #
            # Short conversations become one padded window.
            #

            if len(input_ids) <= self.context_length:

                ids, lbls, mask = self._pad_window(
                    input_ids,
                    labels,
                    loss_mask,
                )

                self.windows.append(
                    TrainingWindow(
                        input_ids=ids,
                        labels=lbls,
                        loss_mask=mask,
                    )
                )

                continue

            #
            # Sliding windows.
            #

            max_start = (
                len(input_ids)
                - self.context_length
            )

            starts = list(
                range(
                    0,
                    max_start + 1,
                    self.stride,
                )
            )

            #
            # Always include the final window.
            #

            if starts[-1] != max_start:
                starts.append(max_start)

            for start in starts:

                end = start + self.context_length

                self.windows.append(
                    TrainingWindow(
                        input_ids=input_ids[start:end],
                        labels=labels[start:end],
                        loss_mask=loss_mask[start:end],
                    )
                )

        print()
        print("=" * 60)
        print("ConversationDataset")
        print("=" * 60)
        print(f"Total Windows : {len(self.windows)}")
        print()

    def __len__(self):

        return len(self.windows)

    def _pad_window(
        self,
        input_ids,
        labels,
        loss_mask,
    ):
        """
        Pads one window to context_length.
        """

        padding = (
            self.context_length
            - len(input_ids)
        )

        if padding <= 0:

            return (
                input_ids,
                labels,
                loss_mask,
            )

        input_ids = (
            input_ids
            + [self.pad_token_id] * padding
        )

        #
        # Ignore padded labels.
        #

        labels = (
            labels
            + [-100] * padding
        )

        #
        # Never compute loss on padding.
        #

        loss_mask = (
            loss_mask
            + [0] * padding
        )

        return (
            input_ids,
            labels,
            loss_mask,
        )

    def __getitem__(
        self,
        index: int,
    ):

        window = self.windows[index]

        return (
            torch.tensor(
                window.input_ids,
                dtype=torch.long,
            ),
            torch.tensor(
                window.labels,
                dtype=torch.long,
            ),
            torch.tensor(
                window.loss_mask,
                dtype=torch.float32,
            ),
        )