class ModelEvaluator:

    def __init__(
        self,
        generator,
    ):
        self.generator = generator

    def evaluate(
        self,
        prompts,
    ):

        print()
        print("=" * 80)
        print("Generation Evaluation")
        print("=" * 80)

        for prompt in prompts:

            print()
            print("-" * 80)
            print("Prompt:")
            print(prompt)
            print()

            output = self.generator.generate(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
            )

            print("Generated:")
            print(output)

        print()