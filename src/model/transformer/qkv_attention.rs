use anyhow::Result;
use tch::{nn::{self, Module}, Tensor};

fn scaled_dot_product_attention(query: Tensor, 
    key: Tensor, value: Tensor, mask: Option<Tensor>) -> Result<(Tensor, Tensor)> {
        let (batch_size, _, _, d_k) = query.size4()?; // 사이즈 구하기

        let k_t = key.transpose(2, 3);
        let mut qk_t = query.matmul(&k_t) / (d_k as f64).sqrt(); // K 전치하고 Q와 내적
    
        if let Some(x) = mask {
            qk_t.masked_fill_(&x, -1e6);
        }

        let attention_score = qk_t.softmax(-1, tch::Kind::Float);

        let output = &attention_score * value;
        
        Ok((output, attention_score))
}