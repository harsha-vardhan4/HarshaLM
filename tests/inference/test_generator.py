# from inference.generator import TextGenerator
# from inference.model_loader import ModelLoader

# from tokenizer.tokenizer import create_tokenizer
# from utils.profiles import development_config


# def test_generate():

#     config = development_config()

#     loader = ModelLoader(config)

#     model = loader.load(
#         "checkpoints/checkpoint_epoch_2.pt"
#     )

#     tokenizer = create_tokenizer()

#     generator = TextGenerator(
#         model,
#         tokenizer,
#         config,
#     )

#     text = generator.generate(
#         prompt="Hello",
#         max_new_tokens=20,
#     )

#     print("\nGenerated Text:\n")
#     print(text)

#     print("\n✓ Generator test passed")


# if __name__ == "__main__":
#     test_generate()

from inference.generator import TextGenerator
from inference.model_loader import ModelLoader

from tokenizer.tokenizer import create_tokenizer
from utils.profiles import development_config


def test_generate():

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

    prompt = "Hello"

    print("=" * 60)
    print("GREEDY")
    print("=" * 60)

    print(
        generator.generate(
            prompt=prompt,
            max_new_tokens=20,
        )
    )

    print("\n")

    print("=" * 60)
    print("TEMPERATURE (0.8)")
    print("=" * 60)

    print(
        generator.generate(
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.8,
        )
    )

    print("\n")

    print("=" * 60)
    print("TOP-K (40)")
    print("=" * 60)

    print(
        generator.generate(
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.8,
            top_k=40,
        )
    )

    print("\n")

    print("=" * 60)
    print("TOP-P (0.9)")
    print("=" * 60)

    print(
        generator.generate(
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.8,
            top_p=0.9,
        )
    )


if __name__ == "__main__":
    test_generate()