from utils.config import ModelConfig

from model.harsha_lm import HarshaLM

from training.optimizer import create_optimizer
from training.scheduler import create_scheduler


def test_scheduler():

    config = ModelConfig()

    model = HarshaLM(config)

    optimizer = create_optimizer(
        model,
        config
    )

    scheduler = create_scheduler(
        optimizer,
        config
    )

    print(type(scheduler))

    for step in range(5):

        scheduler.step()

        print(
            optimizer.param_groups[0]["lr"]
        )

    print("\n✓ Scheduler test passed")


if __name__ == "__main__":
    test_scheduler()