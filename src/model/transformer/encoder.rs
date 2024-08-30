use super::{ffnn::FFNN, multiheadattenion::MultiHeadAttention};
use tch::{nn, Tensor};

struct EncoderBlock {
    attention: MultiHeadAttention,
    ffnn: FFNN,
    norm: nn::LayerNorm,
}

impl EncoderBlock {
    fn new(vs: &nn::Path, d_model: i64, dff: i64, num_heads: i64, drop: f64) -> Self {
        EncoderBlock {
            attention: MultiHeadAttention::new(vs, d_model, num_heads),
            ffnn: FFNN::new(vs, d_model, dff),
            norm: nn::layer_norm(vs, normalized_shape, config),
        }
    }
}
