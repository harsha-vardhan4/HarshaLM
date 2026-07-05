from inference.model_loader import ModelLoader
from utils.profiles import development_config


def test_model_loader():

    config = development_config()

    loader = ModelLoader(config)

    model = loader.load(
        "checkpoints/checkpoint_epoch_2.pt"
    )

    print(type(model))

    print("\n✓ Model Loader test passed")


if __name__ == "__main__":
    test_model_loader()