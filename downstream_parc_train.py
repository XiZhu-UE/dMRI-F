"""Train downstream parc using its adjacent YAML configuration."""
from dimf.downstream.config import load_config


def main():
    from dimf.downstream.training import run_training
    run_training(load_config("parc"))


if __name__ == "__main__":
    main()
