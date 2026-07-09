from training.visualization import (
    TrainingVisualizer,
)


def test_visualization():

    visualizer = (
        TrainingVisualizer(
            log_file="logs/training_metrics.json",
        )
    )

    visualizer.generate()

    print()

    print(
        "✓ Visualization test passed"
    )


if __name__ == "__main__":

    test_visualization()