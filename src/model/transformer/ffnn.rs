use tch::{
    nn::{self, Module},
    Tensor,
};

pub struct FFNN {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl FFNN {
    pub fn new(vs: &nn::Path, d_model: i64, dff: i64) -> Self {
        FFNN {
            fc1: nn::linear(vs, d_model, dff, Default::default()),
            fc2: nn::linear(vs, dff, d_model, Default::default()),
        }
    }

    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        let middle = self.fc1.forward(inputs).relu();
        self.fc2.forward(&middle)
    }
}
