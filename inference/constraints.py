import torch


class GenerationConstraints:
    """
    Constraints applied during text generation.
    """

    @staticmethod
    def no_repeat_ngram(
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        ngram_size: int,
    ) -> torch.Tensor:
        """
        Prevents repeating n-grams.

        Args:

            logits:
                (batch, vocab_size)

            generated_tokens:
                (batch, sequence_length)

            ngram_size:
                Size of forbidden n-gram.

        Returns:

            Modified logits.
        """

        if ngram_size <= 0:
            raise ValueError(
                "ngram_size must be greater than zero."
            )

        logits = logits.clone()

        batch_size = generated_tokens.size(0)

        for batch_index in range(batch_size):

            tokens = generated_tokens[
                batch_index
            ].tolist()

            if len(tokens) < ngram_size:
                continue


            ngram_dict = {}

            for i in range(
                len(tokens) - ngram_size + 1
            ):

                ngram = tuple(
                    tokens[
                        i:i + ngram_size
                    ]
                )

                prefix = ngram[:-1]

                next_token = ngram[-1]


                if prefix not in ngram_dict:

                    ngram_dict[prefix] = []


                ngram_dict[prefix].append(
                    next_token
                )


            current_prefix = tuple(
                tokens[
                    -(ngram_size - 1):
                ]
            )


            banned_tokens = ngram_dict.get(
                current_prefix,
                []
            )


            for token_id in banned_tokens:

                logits[
                    batch_index,
                    token_id
                ] = -float("inf")


        return logits