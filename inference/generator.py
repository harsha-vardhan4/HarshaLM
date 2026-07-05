import torch

from inference.sampling import TokenSampler
from tokenizer.tokenizer import create_tokenizer


class TextGenerator:
    """
    Generates text using a trained HarshaLM model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @torch.no_grad()
    def predict_next_token(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> int:
        """
        Predicts the next token for a given prompt.
        """

        input_ids = self.tokenizer.encode(
            prompt
        )

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.config.device,
        ).unsqueeze(0)

        logits = self.model(
            input_ids
        )

        next_token_logits = logits[
            :,
            -1,
            :
        ]

        next_token = TokenSampler.sample(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return next_token.item()
    
    def _print_top_predictions(
        self,
        logits: torch.Tensor,
        top_n: int = 10,
    ):
        """
        Prints the top predicted tokens.
        """

        values, indices = torch.topk(
            logits,
            k=top_n,
        )

        print("\nTop Predictions")

        for rank in range(top_n):

            token_id = indices[0, rank].item()

            token = self.tokenizer.decode(
                [token_id]
            )

            score = values[0, rank].item()

            print(
                f"{rank+1:2d}. "
                f"id={token_id:<6} "
                f"logit={score:8.4f} "
                f"text={repr(token)}"
            )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Generates text autoregressively.
        """

        input_ids = self.tokenizer.encode(
            prompt
        )

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.config.device,
        ).unsqueeze(0)

        for _ in range(max_new_tokens):

            # Keep only the most recent context window
            if input_ids.size(1) > self.config.context_length:

                input_ids = input_ids[
                    :,
                    -self.config.context_length:
                ]

            logits = self.model(
                input_ids
            )

            next_token_logits = logits[
                :,
                -1,
                :
            ]

            self._print_top_predictions(  # should remove this block of code later
                next_token_logits
            )

            next_token = TokenSampler.sample(
                logits=next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            input_ids = torch.cat(
                (
                    input_ids,
                    next_token,
                ),
                dim=1,
            )

        generated_ids = input_ids.squeeze(0).tolist()

        generated_text = self.tokenizer.decode(
            generated_ids
        )

        return generated_text