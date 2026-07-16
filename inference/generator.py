import torch

from inference.sampling import (
    TokenSampler,
)

from inference.constraints import (
    GenerationConstraints,
)


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
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int | None = None,
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


        if no_repeat_ngram_size is not None:

            next_token_logits = (
                GenerationConstraints.no_repeat_ngram(
                    next_token_logits,
                    input_ids,
                    no_repeat_ngram_size,
                )
            )


        next_token = TokenSampler.sample(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generated_tokens=input_ids,
            repetition_penalty=repetition_penalty,
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


            # print(
            #     f"{rank+1:2d}. "
            #     f"id={token_id:<6} "
            #     f"logit={score:8.4f} "
            #     f"text={repr(token)}"
            # )



    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int | None = None,
        return_full_text: bool = True,
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


        #
        # Save original prompt length.
        #

        prompt_length = input_ids.size(1)



        #
        # Tokens that stop generation.
        #

        stop_tokens = {
            self.tokenizer.eos_token_id,
        }


        if hasattr(
            self.tokenizer,
            "user_token_id",
        ):

            stop_tokens.add(
                self.tokenizer.user_token_id
            )


        if hasattr(
            self.tokenizer,
            "system_token_id",
        ):

            stop_tokens.add(
                self.tokenizer.system_token_id
            )


        generated_tokens = 0

        for _ in range(max_new_tokens):

            if input_ids.size(1) > self.config.context_length:

                input_ids = input_ids[
                    :,
                    -self.config.context_length:
                ]

                prompt_length = input_ids.size(1)



            logits = self.model(
                input_ids
            )


            next_token_logits = logits[
                :,
                -1,
                :
            ]

            if no_repeat_ngram_size is not None:

                next_token_logits = (
                    GenerationConstraints.no_repeat_ngram(
                        next_token_logits,
                        input_ids,
                        no_repeat_ngram_size,
                    )
                )

            next_token = TokenSampler.sample(
                logits=next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_tokens=input_ids,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )



            input_ids = torch.cat(
                (
                    input_ids,
                    next_token,
                ),
                dim=1,
            )
            generated_tokens += 1



            #
            # Stop generation.
            #

            if (
                next_token.item() in stop_tokens
                and generated_tokens > 5
            ):
                break

        generated_ids = (
            input_ids
            .squeeze(0)
            .tolist()
        )

        #
        # Return complete prompt + response.
        #

        if return_full_text:

            return self.tokenizer.decode(
                generated_ids
            )

        #
        # Calculate completion safely.
        #

        effective_prompt_length = min(
            prompt_length,
            len(generated_ids),
        )


        completion_ids = generated_ids[
            effective_prompt_length:
        ]



        #
        # Debug
        #

        # print()

        # print("=" * 60)
        # print("Generation Debug")
        # print("-" * 60)

        # print(
        #     f"Prompt Tokens   : {prompt_length}"
        # )

        # print(
        #     f"Generated Tokens: {len(generated_ids)}"
        # )

        # print(
        #     f"Completion IDs  : {completion_ids}"
        # )


        # decoded = self.tokenizer.decode(
        #     completion_ids
        # )


        # print(
        #     f"Decoded         : {repr(decoded)}"
        # )

        # print("=" * 60)

        # print()



        # return decoded.strip()

        return self.tokenizer.decode(
            completion_ids
        ).strip()