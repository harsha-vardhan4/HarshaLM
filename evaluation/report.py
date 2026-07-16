from evaluation.evaluator import EvaluationResult


class EvaluationReport:
    """
    Prints evaluation results.
    """

    def __init__(
        self,
        results: list[EvaluationResult],
    ):

        self.results = results

    def print(self):
        """
        Prints all evaluation results.
        """

        print()
        print("=" * 80)
        print("HarshaLM Evaluation Report")
        print("=" * 80)

        current_category = None

        for result in self.results:

            #
            # Print category header once
            #

            if result.category != current_category:

                current_category = result.category

                print()
                print("-" * 80)
                print(current_category)
                print("-" * 80)

            print()

            print(
                f"Prompt Name : {result.name}"
            )

            print(
                f"Prompt      : {result.prompt}"
            )

            print()

            print("Generated")

            print(result.generated_text)

            print()

            print("-" * 80)

        print()

        print("=" * 80)

        print(
            f"Evaluation Complete ({len(self.results)} prompts)"
        )

        print("=" * 80)

        print()