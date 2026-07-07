from model.harsha_lm import HarshaLM
from utils.config import ModelConfig
from utils.parameter_summary import ParameterSummary


def test_parameter_summary():

    config = ModelConfig()

    model = HarshaLM(config)

    ParameterSummary.print(model)

    print("\n✓ Parameter summary test passed")


if __name__ == "__main__":
    test_parameter_summary()