import subprocess
import sys


TESTS = [

    #
    # Attention
    #

    "tests.attention.test_multi_head_attention",
    "tests.attention.test_scaled_dot_product",

    #
    # Data
    #

    "tests.data.test_loader",

    #
    # embeddings
    #

    "tests.embeddings.test_embedding_layer",
    "tests.embeddings.test_positional_embedding",
    "tests.embeddings.test_token_embedding",

    #
    # heads
    #

    "tests.heads.test_lm_head",



    #
    # Tokenizer
    #

    "tests.tokenizer.test_tokenizer",

    #
    # Model
    #

    "tests.model.test_harsha_lm",

    #
    # Transformer
    #

    "tests.transformer.test_feed_forward",

    "tests.transformer.test_transformer_block",

    "tests.transformer.test_transformer_stack",

    #
    # Training
    #

    "tests.training.test_dataloader",

    "tests.training.test_train_batch",

    "tests.training.test_train_epoch",

    "tests.training.test_components",

    "tests.training.test_pipeline",

    "tests.training.test_trainer",

    "tests.training.test_checkpoint",

    "tests.training.test_optimizer",

    "tests.training.test_scheduler",

    "tests.training.test_loss",

    "tests.training.test_dataset",

    #
    # Inference
    #

    "tests.inference.test_model_loader",

    "tests.inference.test_logits",

    "tests.inference.test_generator",

    "tests.inference.test_sampling",

    "tests.inference.test_top_k",

    "tests.inference.test_top_p",


    #
    # test
    #

    "tests.test",

]


def run_test(module):

    print()

    print("=" * 70)

    print(module)

    print("=" * 70)

    result = subprocess.run(

        [
            sys.executable,
            "-m",
            module,
        ]

    )

    return result.returncode == 0


def main():

    passed = 0

    failed = 0

    failures = []

    for test in TESTS:

        success = run_test(test)

        if success:

            passed += 1

        else:

            failed += 1

            failures.append(test)

    print()

    print("=" * 70)

    print("SUMMARY")

    print("=" * 70)

    print(f"Passed : {passed}")

    print(f"Failed : {failed}")

    if failures:

        print()

        print("Failed Tests:")

        for failure in failures:

            print(f" - {failure}")

    else:

        print()

        print("🎉 All tests passed!")



if __name__ == "__main__":
    main()