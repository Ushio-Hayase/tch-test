use anyhow::Result;
use tch::{nn::{self, batch_norm2d, BatchNorm, Module}, utils};

struct CNN {
    norm: nn::BatchNorm,
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D
}

impl CNN {
    fn new(vs: &nn::Path) -> Self {
        CNN {
            norm: batch_norm2d(vs, out_dim, config)
        }
    }
}