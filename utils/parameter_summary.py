from collections import OrderedDict

from model.harsha_lm import HarshaLM


class ParameterSummary:
    """
    Prints a parameter breakdown for the model.

    The summary is implementation-independent and
    automatically discovers every parameter using
    model.named_parameters().
    """

    @staticmethod
    def print(model: HarshaLM):

        print()
        print("=" * 80)
        print("HarshaLM Parameter Breakdown")
        print("=" * 80)

        grouped_parameters = OrderedDict()

        total_parameters = 0
        trainable_parameters = 0

        #
        # Collect parameters
        #

        for name, parameter in model.named_parameters():

            parameter_count = parameter.numel()

            total_parameters += parameter_count

            if parameter.requires_grad:
                trainable_parameters += parameter_count

            #
            # Group by the first two module names.
            #
            # Example:
            #
            # embedding_layer.token_embedding.embedding.weight
            # -> embedding_layer.token_embedding
            #
            # transformer.blocks.0.attention.q_projection.weight
            # -> transformer.blocks
            #

            parts = name.split(".")

            if len(parts) >= 2:
                group = ".".join(parts[:2])
            else:
                group = parts[0]

            if group not in grouped_parameters:
                grouped_parameters[group] = []

            grouped_parameters[group].append(
                (
                    name,
                    tuple(parameter.shape),
                    parameter_count,
                )
            )

        #
        # Print groups
        #

        for group_name, parameters in grouped_parameters.items():

            print()
            print(group_name)

            print("-" * 80)

            group_total = 0

            for name, shape, count in parameters:

                group_total += count

                print(
                    f"{name:<60}"
                    f"{str(shape):<20}"
                    f"{count:>12,}"
                )

            print("-" * 80)

            print(
                f"{'Group Total':<80}"
                f"{group_total:>12,}"
            )

        #
        # Weight tying check
        #

        print()
        print("=" * 80)

        if model.verify_weight_tying():

            print("Weight Tying : Enabled")

        else:

            print("Weight Tying : Disabled")

        non_trainable = (
            total_parameters -
            trainable_parameters
        )

        model_size_mb = sum(

            parameter.numel() *
            parameter.element_size()

            for parameter in model.parameters()

        ) / (1024 ** 2)

        print()

        print(
            f"{'Total Parameters':35}"
            f"{total_parameters:>15,}"
        )

        print(
            f"{'Trainable Parameters':35}"
            f"{trainable_parameters:>15,}"
        )

        print(
            f"{'Non-Trainable Parameters':35}"
            f"{non_trainable:>15,}"
        )

        print(
            f"{'Approx Model Size':35}"
            f"{model_size_mb:>14.2f} MB"
        )

        print("=" * 80)