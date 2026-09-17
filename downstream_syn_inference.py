"""Run T1 synthesis using downstream_syn_inference.ymal."""
from dimf.downstream.inference_config import load_inference_config
from dimf.downstream.inference import run_inference


if __name__ == '__main__':
    run_inference(load_inference_config('syn'))
