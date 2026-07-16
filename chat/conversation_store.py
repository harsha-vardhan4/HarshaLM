import json

from pathlib import Path

from datetime import datetime



class ConversationStore:
    """
    Saves conversations for future training.
    """


    def __init__(
        self,
        directory="datasets/sharegpt",
    ):

        self.directory = Path(
            directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )



    def save(
        self,
        messages,
    ):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


        conversations = []


        for message in messages:

            role = message["role"]

            conversations.append(
                {
                    "from":
                        "human"
                        if role == "user"
                        else "gpt",

                    "value":
                        message["content"],
                }
            )


        data = {
            "id": timestamp,
            "conversations": conversations,
        }


        file_path = (
            self.directory
            /
            f"conversation_{timestamp}.json"
        )


        with open(
            file_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )