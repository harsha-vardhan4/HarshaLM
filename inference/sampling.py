import torch


class TokenSampler:
    """
    Sampling strategies used during text generation.
    """

    @staticmethod
    def greedy(
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Greedy decoding.
        """

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
        """
        Temperature sampling.
        """

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
    def top_k(
        logits: torch.Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Top-k sampling.
        """

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        # Scale logits
        logits = logits / temperature

        # Get top-k logits and indices
        top_k_logits, top_k_indices = torch.topk(
            logits,
            k,
            dim=-1,
        )

        # Convert top-k logits to probabilities
        probabilities = torch.softmax(
            top_k_logits,
            dim=-1,
        )

        # Sample within top-k
        sampled_index = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # Convert sampled position back to original vocabulary index
        next_token = torch.gather(
            top_k_indices,
            dim=-1,
            index=sampled_index,
        )

        return next_token
    
    @staticmethod
    def top_p(
        logits: torch.Tensor,
        p: float,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Top-p (Nucleus) sampling.
        """

        if not (0 < p <= 1):
            raise ValueError(
                "p must be in (0, 1]."
            )

        # Scale logits
        logits = logits / temperature

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=-1,
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        # Remove tokens whose cumulative probability exceeds p
        sorted_mask = cumulative_probs > p

        # Keep the first token above the threshold
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        sorted_probs[sorted_mask] = 0

        # Renormalize
        sorted_probs = sorted_probs / sorted_probs.sum(
            dim=-1,
            keepdim=True,
        )

        sampled_index = torch.multinomial(
            sorted_probs,
            num_samples=1,
        )

        next_token = torch.gather(
            sorted_indices,
            dim=-1,
            index=sampled_index,
        )

        return next_token

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:

        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        if top_k is not None and top_k <= 0:
            raise ValueError(
                "top_k must be > 0"
            )

        if top_p is not None and not (0 < top_p <= 1):
            raise ValueError(
                "top_p must be in (0, 1]"
            )

        if top_p is not None:

            return TokenSampler.top_p(
                logits,
                p=top_p,
                temperature=temperature,
            )

        if top_k is not None:

            return TokenSampler.top_k(
                logits,
                k=top_k,
                temperature=temperature,
            )

        if temperature == 1.0:

            return TokenSampler.greedy(
                logits
            )

        return TokenSampler.temperature(
            logits,
            temperature,
        )