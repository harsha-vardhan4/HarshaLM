from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationPrompt:
    """
    Represents one evaluation prompt.

    Attributes
    ----------
    category:
        Category of the prompt.

    name:
        Short descriptive name.

    prompt:
        Text given to HarshaLM.
    """

    category: str
    name: str
    prompt: str


EVALUATION_PROMPTS = [

    #
    # Greetings
    #

    EvaluationPrompt(
        category="Greeting",
        name="Simple Hello",
        prompt="Hello!"
    ),

    EvaluationPrompt(
        category="Greeting",
        name="Good Morning",
        prompt="Good morning."
    ),

    #
    # Conversation
    #

    EvaluationPrompt(
        category="Conversation",
        name="How are you",
        prompt="How are you?"
    ),

    EvaluationPrompt(
        category="Conversation",
        name="Introduce Yourself",
        prompt="Tell me about yourself."
    ),

    #
    # Question Answering
    #

    EvaluationPrompt(
        category="Question",
        name="Capital",
        prompt="What is the capital of France?"
    ),

    EvaluationPrompt(
        category="Question",
        name="Programming",
        prompt="What is Python?"
    ),

    #
    # Completion
    #

    EvaluationPrompt(
        category="Completion",
        name="Sentence",
        prompt="Artificial Intelligence is"
    ),

    EvaluationPrompt(
        category="Completion",
        name="Programming",
        prompt="Machine learning allows computers to"
    ),

    #
    # Story
    #

    EvaluationPrompt(
        category="Story",
        name="Adventure",
        prompt="Once upon a time"
    ),

    #
    # Reasoning
    #

    EvaluationPrompt(
        category="Reasoning",
        name="Simple Logic",
        prompt="If today is Monday, tomorrow is"
    ),

    #
    # Mathematics
    #

    EvaluationPrompt(
        category="Math",
        name="Addition",
        prompt="2 + 2 ="
    ),

    #
    # Coding
    #

    EvaluationPrompt(
        category="Code",
        name="Python Function",
        prompt="Write a Python function to"
    ),

]