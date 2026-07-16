from data.loaders.loader import TextDatasetLoader


def test_loader():

    loader = TextDatasetLoader()

    #
    # Single file
    #

    text = loader.load(
        "datasets/sample.txt"
    )

    print()

    print("Single File")

    print(f"Characters : {len(text)}")

    print()

    print(text[:200])

    assert isinstance(text, str)
    assert len(text) > 0

    #
    # Directory
    #

    text = loader.load(
        "datasets"
    )

    print()

    print("Directory")

    print(f"Characters : {len(text)}")

    assert isinstance(text, str)
    assert len(text) > 0

    print()

    print("✓ Loader test passed")


if __name__ == "__main__":
    test_loader()