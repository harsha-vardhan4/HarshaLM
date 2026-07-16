from chat.chat_session import (
    ChatSession,
)

from chat.conversation_store import (
    ConversationStore,
)



class ChatCLI:
    """
    Command-line interface for chatting with HarshaLM.
    """



    def __init__(
        self,
        session: ChatSession,
    ):

        self.session = session

        self.store = ConversationStore()



    def run(
        self,
    ) -> None:
        """
        Starts interactive chat.
        """


        print()

        print("=" * 60)
        print("HarshaLM Chat")
        print("-" * 60)
        print("Type 'exit' to quit.")
        print("Type 'clear' to clear the conversation.")
        print("=" * 60)

        print()



        while True:


            try:

                user_input = input(
                    "You: "
                ).strip()


            except (
                KeyboardInterrupt,
                EOFError,
            ):


                print()

                self.save()

                print(
                    "Goodbye!"
                )

                break



            if not user_input:

                continue



            if user_input.lower() == "exit":


                print()

                self.save()

                print(
                    "Goodbye!"
                )

                break



            if user_input.lower() == "clear":


                self.session.clear()


                print()

                print(
                    "Conversation cleared."
                )

                print()

                continue



            reply = self.session.chat(
                user_input
            )


            print()

            print(
                f"HarshaLM: {reply}"
            )


            print()


            edit = input(
                "Edit response? (y/n): "
            ).strip().lower()



            if edit == "y":


                corrected = input(
                    "Correct response: "
                ).strip()


                self.session.update_last_response(
                    corrected
                )


                print(
                    "Response updated."
                )


            print()



    def save(
        self,
    ):

        messages = (
            self.session.get_messages()
        )


        if messages:

            self.store.save(
                messages
            )