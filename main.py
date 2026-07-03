from utils.tokenizer import Tokenizer


def main():

    tokenizer = Tokenizer(
        "data/tokenizer/mytokenizer.model"
    )

    print("=" * 50)
    print("HarshaLM Tokenizer")
    print("=" * 50)

    print()

    sentence = "Hello, I am building HarshaLM."

    tokens = tokenizer.encode(sentence)

    print("Original Sentence:")
    print(sentence)

    print()

    print("Token IDs:")
    print(tokens)

    print()

    decoded = tokenizer.decode(tokens)

    print("Decoded Sentence:")
    print(decoded)

    print()

    print("Vocabulary Size:", tokenizer.vocab_size)
    print("PAD:", tokenizer.pad_id)
    print("BOS:", tokenizer.bos_id)
    print("EOS:", tokenizer.eos_id)
    print("UNK:", tokenizer.unk_id)


if __name__ == "__main__":
    main()