import torch

from inference.sampling import TokenSampler


def test_greedy():

    logits = torch.tensor(
        [
            [
                1.0,
                5.0,
                2.0,
                3.0,
            ]
        ]
    )

    token = TokenSampler.greedy(
        logits
    )

    assert token.item() == 1

    print("✓ Greedy sampling passed")



def test_temperature():

    torch.manual_seed(42)

    logits = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        ]
    )


    token = TokenSampler.temperature(
        logits,
        temperature=0.8,
    )


    assert token.shape == (
        1,
        1,
    )


    assert 0 <= token.item() < 4


    print("✓ Temperature sampling passed")



def test_top_k():

    torch.manual_seed(42)


    logits = torch.tensor(
        [
            [
                10.0,
                9.0,
                8.0,
                1.0,
                0.0,
            ]
        ]
    )


    token = TokenSampler.top_k(
        logits,
        k=3,
    )


    # token must be inside:
    #
    # index 0,1,2
    #

    assert token.item() in [
        0,
        1,
        2,
    ]


    print("✓ Top-k sampling passed")



def test_top_p():

    torch.manual_seed(42)


    logits = torch.tensor(
        [
            [
                10.0,
                9.0,
                1.0,
                0.5,
                0.1,
            ]
        ]
    )


    token = TokenSampler.top_p(
        logits,
        p=0.9,
    )


    assert token.shape == (
        1,
        1,
    )


    assert 0 <= token.item() < 5


    print("✓ Top-p sampling passed")



def test_repetition_penalty():

    logits = torch.tensor(
        [
            [
                5.0,
                4.0,
                3.0,
                2.0,
            ]
        ]
    )


    generated_tokens = torch.tensor(
        [
            [
                0,
                2,
            ]
        ]
    )


    modified_logits = (
        TokenSampler.repetition_penalty(
            logits,
            generated_tokens,
            penalty=2.0,
        )
    )


    # Token 0:
    #
    # 5 / 2 = 2.5
    #

    assert modified_logits[
        0,
        0,
    ] == 2.5



    # Token 2:
    #
    # 3 / 2 = 1.5
    #

    assert modified_logits[
        0,
        2,
    ] == 1.5


    # Token 1 untouched

    assert modified_logits[
        0,
        1,
    ] == 4.0


    print("✓ Repetition penalty passed")



def test_invalid_temperature():

    logits = torch.randn(
        1,
        10,
    )


    try:

        TokenSampler.sample(
            logits,
            temperature=0,
        )

        assert False


    except ValueError:

        pass


    print(
        "✓ Temperature validation passed"
    )



def test_invalid_top_k():

    logits = torch.randn(
        1,
        10,
    )


    try:

        TokenSampler.top_k(
            logits,
            k=0,
        )

        assert False


    except ValueError:

        pass


    print(
        "✓ Top-k validation passed"
    )



def test_invalid_top_p():

    logits = torch.randn(
        1,
        10,
    )


    try:

        TokenSampler.top_p(
            logits,
            p=1.5,
        )

        assert False


    except ValueError:

        pass


    print(
        "✓ Top-p validation passed"
    )



if __name__ == "__main__":

    test_greedy()

    test_temperature()

    test_top_k()

    test_top_p()

    test_repetition_penalty()

    test_invalid_temperature()

    test_invalid_top_k()

    test_invalid_top_p()


    print(
        "\n✓ All sampling tests passed"
    )