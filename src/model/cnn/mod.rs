use anyhow::Result;
use tch::{nn::{self, Module}, utils};

struct CNN {
    norm: nn::BatchNorm,
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D
}

impl CNN {
    fn new(vs: &nn::Path) -> Self {
        CNN {
            norm: 
        }
    }
}