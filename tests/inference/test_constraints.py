import torch

from inference.constraints import GenerationConstraints


def test_no_repeat_ngram():

    """
    Tests that previously generated n-grams
    are blocked during generation.
    """

    vocab_size = 10

    logits = torch.zeros(
        1,
        vocab_size,
    )


    # Generated sequence:
    #
    # [1, 2, 3, 1, 2]
    #
    # With ngram_size=3:
    #
    # Existing ngram:
    # (1,2,3)
    #
    # Current prefix:
    # (1,2)
    #
    # Token 3 should be blocked.
    

    generated_tokens = torch.tensor(
        [
            [
                1,
                2,
                3,
                1,
                2,
            ]
        ],
        dtype=torch.long,
    )


    modified_logits = (
        GenerationConstraints.no_repeat_ngram(
            logits,
            generated_tokens,
            ngram_size=3,
        )
    )


    banned_token_score = (
        modified_logits[
            0,
            3,
        ]
    )


    assert banned_token_score == -float("inf")


    print(
        "Blocked token:",
        banned_token_score.item(),
    )


    print(
        "\n✓ No-repeat n-gram test passed"
    )



def test_no_repeat_ngram_no_match():

    """
    Tests that unrelated tokens
    are not blocked.
    """

    logits = torch.zeros(
        1,
        10,
    )


    generated_tokens = torch.tensor(
        [
            [
                1,
                2,
                4,
                5,
            ]
        ],
        dtype=torch.long,
    )


    modified_logits = (
        GenerationConstraints.no_repeat_ngram(
            logits,
            generated_tokens,
            ngram_size=3,
        )
    )


    assert torch.all(
        modified_logits != -float("inf")
    )


    print(
        "\n✓ No-repeat n-gram no-match test passed"
    )



if __name__ == "__main__":

    test_no_repeat_ngram()

    test_no_repeat_ngram_no_match()