use super::qkv_attention::scaled_dot_product_attention;
use anyhow::{Error, Ok, Result};
use tch::{
    nn::{self, Module},
    Tensor,
};

pub struct MultiHeadAttention {
    wq: nn::Linear,
    wk: nn::Linear,
    wv: nn::Linear,
    out_fc: nn::Linear,
    num_heads: i64,
    d_k: i64,
}

impl MultiHeadAttention {
    pub fn new(vs: &nn::Path, d_model: i64, num_heads: i64) -> Self {
        if d_model % num_heads != 0 {
            panic!("Cannot divide d_model to num_heads");
        }
        let d_k: i64 = d_model / num_heads;

        MultiHeadAttention {
            wq: nn::linear(vs, d_model, d_k, Default::default()),
            wk: nn::linear(vs, d_model, d_k, Default::default()),
            wv: nn::linear(vs, d_model, d_k, Default::default()),
            out_fc: nn::linear(vs, d_model, d_k, Default::default()),
            num_heads: num_heads,
            d_k: d_k,
        }
    }

    fn split_heads(&self, inputs: &Tensor, batch_size: i64) -> Tensor {
        inputs.view_([batch_size, -1, self.num_heads, self.d_k]);
        return inputs.permute([0, 2, 1, 3]);
    }

    fn concat(inputs: &Tensor) -> Tensor {
        let (batch_size, _, seq_len, d_k) = inputs
            .size4()
            .expect("inputs shape length is not 4 in MultiHeadAttention concat");

        inputs
            .transpose(1, 2)
            .contiguous()
            .view([batch_size, seq_len, d_k])
    }

    pub fn forward(
        self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let (batch_size, _, _) = query
            .size3()
            .expect("inputs shape length is not 3 in MultiHeadAttention");

        // 차원 변환
        let trans_query = &self.wq.forward(query);
        let trans_key = &self.wk.forward(key);
        let trans_value = &self.wv.forward(value);

        // num heads에 따라 텐서 분리
        let split_query = self.split_heads(trans_query, batch_size);
        let split_key = self.split_heads(trans_key, batch_size);
        let split_value = self.split_heads(trans_value, batch_size);

        let (out, attention_score) =
            scaled_dot_product_attention(&split_query, &split_key, &split_value, mask);

        let out = self.out_fc.forward(&out);

        out
    }
}
