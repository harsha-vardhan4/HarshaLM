import torch


class TokenSampler:
    """
    Sampling strategies used during text generation.
    """


    @staticmethod
    def greedy(
        logits: torch.Tensor,
    ) -> torch.Tensor:

        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )


    @staticmethod
    def temperature(
        logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:

        logits = logits / temperature

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        return torch.multinomial(
            probabilities,
            num_samples=1,
        )


    @staticmethod
    def repetition_penalty(
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:

        if penalty <= 0:
            raise ValueError(
                "penalty must be greater than zero."
            )


        logits = logits.clone()


        for batch_index in range(
            logits.size(0)
        ):

            previous_tokens = set(
                generated_tokens[
                    batch_index
                ].tolist()
            )


            for token_id in previous_tokens:

                if logits[
                    batch_index,
                    token_id
                ] > 0:

                    logits[
                        batch_index,
                        token_id
                    ] /= penalty

                else:

                    logits[
                        batch_index,
                        token_id
                    ] *= penalty


        return logits



    @staticmethod
    def top_k(
        logits: torch.Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:

        if k <= 0:

            raise ValueError(
                "k must be greater than zero."
            )


        if k > logits.size(-1):

            raise ValueError(
                "k cannot exceed vocabulary size."
            )


        logits = logits / temperature


        top_k_logits, top_k_indices = torch.topk(
            logits,
            k,
            dim=-1,
        )


        probabilities = torch.softmax(
            top_k_logits,
            dim=-1,
        )


        sampled_index = torch.multinomial(
            probabilities,
            num_samples=1,
        )


        return torch.gather(
            top_k_indices,
            dim=-1,
            index=sampled_index,
        )



    @staticmethod
    def top_p(
        logits: torch.Tensor,
        p: float,
        temperature: float = 1.0,
    ) -> torch.Tensor:


        if not (0 < p <= 1):

            raise ValueError(
                "p must be between 0 and 1."
            )


        logits = logits / temperature


        probabilities = torch.softmax(
            logits,
            dim=-1,
        )


        sorted_probs, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=-1,
        )


        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1,
        )


        sorted_mask = cumulative_probs > p


        sorted_mask[...,1:] = (
            sorted_mask[...,:-1]
            .clone()
        )

        sorted_mask[...,0] = False


        sorted_probs[sorted_mask] = 0


        sorted_probs = (
            sorted_probs /
            sorted_probs.sum(
                dim=-1,
                keepdim=True,
            )
        )


        sampled_index = torch.multinomial(
            sorted_probs,
            num_samples=1,
        )


        return torch.gather(
            sorted_indices,
            dim=-1,
            index=sampled_index,
        )



    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generated_tokens: torch.Tensor | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:


        if temperature <= 0:

            raise ValueError(
                "temperature must be greater than zero."
            )


        if generated_tokens is not None:

            if repetition_penalty != 1.0:

                logits = TokenSampler.repetition_penalty(
                    logits,
                    generated_tokens,
                    repetition_penalty,
                )


        if top_p is not None:

            return TokenSampler.top_p(
                logits,
                top_p,
                temperature,
            )


        if top_k is not None:

            return TokenSampler.top_k(
                logits,
                top_k,
                temperature,
            )


        if temperature == 1.0:

            return TokenSampler.greedy(
                logits
            )


        return TokenSampler.temperature(
            logits,
            temperature,
        )