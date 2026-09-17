"""Run using the adjacent dataset.ymal; no command-line arguments are needed."""
from dimf.config import load_config

def main():
    config = load_config('dataset')
    from dimf.data.conversion import run_nii2npy
    run_nii2npy(config)

if __name__ == '__main__':
    main()
