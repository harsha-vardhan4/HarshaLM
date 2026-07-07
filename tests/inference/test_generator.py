from inference.generator import TextGenerator
from inference.model_loader import ModelLoader

from tokenizer.tokenizer import create_tokenizer
from utils.profiles import development_config


def test_generate():

    config = development_config()

    loader = ModelLoader(config)

    model = loader.load()
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

    greedy_text = generator.generate(
        prompt=prompt,
        max_new_tokens=20,
    )

    print(greedy_text)



    print("\n")
    print("=" * 60)
    print("TEMPERATURE (0.8)")
    print("=" * 60)

    temperature_text = generator.generate(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,
    )

    print(temperature_text)



    print("\n")
    print("=" * 60)
    print("TOP-K (40)")
    print("=" * 60)

    top_k_text = generator.generate(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=40,
    )

    print(top_k_text)



    print("\n")
    print("=" * 60)
    print("TOP-P (0.9)")
    print("=" * 60)

    top_p_text = generator.generate(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_p=0.9,
    )

    print(top_p_text)



    print("\n")
    print("=" * 60)
    print("REPETITION PENALTY (1.2)")
    print("=" * 60)

    repetition_text = generator.generate(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,
        repetition_penalty=1.2,
    )

    print(repetition_text)



    print("\n")
    print("=" * 60)
    print("NO REPEAT N-GRAM (3)")
    print("=" * 60)

    ngram_text = generator.generate(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,
        no_repeat_ngram_size=3,
    )

    print(ngram_text)



    # Basic validations

    assert isinstance(
        greedy_text,
        str,
    )

    assert len(
        greedy_text
    ) > 0


    assert isinstance(
        repetition_text,
        str,
    )

    assert isinstance(
        ngram_text,
        str,
    )


    print("\n✓ Generator test passed")


if __name__ == "__main__":
    test_generate()