from dataclasses import dataclass

from inference.generator import TextGenerator

from evaluation.prompts import (
    EVALUATION_PROMPTS,
    EvaluationPrompt,
)


@dataclass
class EvaluationResult:
    """
    Stores one evaluation result.
    """

    category: str

    name: str

    prompt: str

    generated_text: str


class Evaluator:
    """
    Runs HarshaLM on a fixed benchmark.
    """

    def __init__(
        self,
        generator: TextGenerator,
    ):

        self.generator = generator

    def evaluate_prompt(
        self,
        prompt: EvaluationPrompt,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int | None = None,
    ) -> EvaluationResult:
        """
        Evaluates a single prompt.
        """

        generated = self.generator.generate(
            prompt=prompt.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        return EvaluationResult(
            category=prompt.category,
            name=prompt.name,
            prompt=prompt.prompt,
            generated_text=generated,
        )

    def evaluate(
        self,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int | None = None,
    ) -> list[EvaluationResult]:
        """
        Evaluates all benchmark prompts.
        """

        results: list[EvaluationResult] = []

        for prompt in EVALUATION_PROMPTS:

            result = self.evaluate_prompt(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            results.append(result)

        return results