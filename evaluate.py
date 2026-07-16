import torch

from utils.config import ModelConfig

from model.harsha_lm import HarshaLM

from tokenizer.tokenizer import create_tokenizer

from inference.generator import TextGenerator

from training.checkpoint import CheckpointManager

from evaluation.evaluator import Evaluator
from evaluation.report import EvaluationReport


def main():

    #
    # Configuration
    #

    config = ModelConfig()

    device = torch.device(config.device)

    #
    # Tokenizer
    #

    tokenizer = create_tokenizer()

    config.vocab_size = tokenizer.vocab_size

    #
    # Model
    #

    model = HarshaLM(config)

    model.to(device)

    #
    # Load best checkpoint
    #

    checkpoint_manager = CheckpointManager(config)

    checkpoint_manager.load(
        config.resume_checkpoint,
        model,
    )

    model.eval()

    #
    # Text Generator
    #

    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    #
    # Evaluator
    #

    evaluator = Evaluator(
        generator=generator,
    )

    results = evaluator.evaluate(
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    #
    # Report
    #

    report = EvaluationReport(results)

    report.print()


if __name__ == "__main__":

    main()