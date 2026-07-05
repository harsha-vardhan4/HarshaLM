from inference.generator import TextGenerator
from inference.model_loader import ModelLoader

from tokenizer.tokenizer import create_tokenizer
from utils.profiles import development_config


def test_logits():

    config = development_config()

    loader = ModelLoader(config)

    model = loader.load(
        "checkpoints/checkpoint_epoch_2.pt"
    )

    tokenizer = create_tokenizer()

    generator = TextGenerator(
        model,
        tokenizer,
        config,
    )

    generator.generate(
        prompt="Hello",
        max_new_tokens=1,
    )

    print(
        "\n✓ Logit inspection test passed"
    )


if __name__ == "__main__":
    test_logits()