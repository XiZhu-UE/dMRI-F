"""Run using the adjacent dataset.ymal; no command-line arguments are needed."""
from dimf.config import load_config

def main():
    config = load_config('dataset')
    from dimf.data.shells import run_shell_max
    run_shell_max(config)

if __name__ == '__main__':
    main()
