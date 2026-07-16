from chat.chat_session import (
    ChatSession,
)

from chat.cli import (
    ChatCLI,
)

from inference.generator import (
    TextGenerator,
)

from inference.model_loader import (
    ModelLoader,
)

from tokenizer.tokenizer import (
    create_tokenizer,
)

from utils.profiles import (
    development_config,
)


def main():
    """
    Starts the HarshaLM interactive chat.
    """

    #
    # Load configuration
    #

    config = development_config()

    #
    # Load model
    #

    loader = ModelLoader(
        config
    )

    model = loader.load()

    #
    # Load tokenizer
    #

    tokenizer = create_tokenizer()

    #
    # Create text generator
    #

    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    #
    # Create chat session
    #

    session = ChatSession(
        generator=generator,
    )

    #
    # Start CLI
    #

    cli = ChatCLI(
        session=session,
    )

    cli.run()


if __name__ == "__main__":

    main()