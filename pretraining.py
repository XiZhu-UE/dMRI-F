"""Run using the adjacent pretraining.ymal; no command-line arguments are needed."""
from dimf.config import load_config

def main():
    config = load_config('pretraining')
    from dimf.training import run_training
    run_training(config)

if __name__ == '__main__':
    main()
