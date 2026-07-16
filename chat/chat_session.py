from inference.generator import (
    TextGenerator,
)


class ChatSession:
    """
    Manages a single HarshaLM conversation session.
    """


    def __init__(
        self,
        generator: TextGenerator,
    ):

        self.generator = generator

        self.messages = []



    def build_prompt(
        self,
    ) -> str:
        """
        Builds conversation prompt
        from history.
        """

        prompt = ""

        for message in self.messages:

            role = message["role"]

            content = message["content"]


            if role == "user":

                prompt += (
                    f"User: {content}\n"
                )


            elif role == "assistant":

                prompt += (
                    f"Assistant: {content}\n"
                )


        prompt += "Assistant: "

        return prompt



    def chat(
        self,
        user_input: str,
    ) -> str:
        """
        Generates assistant response.
        """


        self.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )


        prompt = self.build_prompt()


        reply = self.generator.generate(
            prompt=prompt,
            max_new_tokens=40,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            return_full_text=False,
        )


        self.messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )


        return reply



    def update_last_response(
        self,
        new_response: str,
    ):
        """
        Updates the latest assistant message.
        """


        if not self.messages:

            return


        if (
            self.messages[-1]["role"]
            ==
            "assistant"
        ):

            self.messages[-1][
                "content"
            ] = new_response



    def clear(
        self,
    ):
        """
        Clears current conversation.
        """

        self.messages = []



    def get_messages(
        self,
    ):
        """
        Returns current conversation.
        """

        return self.messages