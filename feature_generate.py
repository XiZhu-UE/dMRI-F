"""Example: create a 64-channel feature NIfTI from b0 + six DWI volumes."""
from dimf.features import load_config, run


if __name__ == '__main__':
    run(load_config())
