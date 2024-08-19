use anyhow::{Result, Error};
use tch::{nn::{self, Module}, Tensor};

struct MultiHeadAttention {
    wq: nn::Linear,
    wk: nn::Linear,
    wv: nn::Linear,
    out_fc: nn::Linear
}

impl MultiHeadAttention {
    fn new(d_model: u32, num_heads: u32) -> Result<Tensor, Error> {
        if(d_model % num_heads != 0) {
            Err()
        }
    }
}
