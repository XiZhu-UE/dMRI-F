"""Run using the adjacent inference.ymal; no command-line arguments are needed."""
from dimf.config import load_config

def main():
    config = load_config('inference')
    from dimf.inference import run_inference
    run_inference(config)

if __name__ == '__main__':
    main()
