"""Train downstream syn using its adjacent YAML configuration."""
from dimf.downstream.config import load_config


def main():
    from dimf.downstream.training import run_training
    run_training(load_config("syn"))


if __name__ == "__main__":
    main()
