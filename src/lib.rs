mod model;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        model::vae::run(1e-3, 10, 128).unwrap();
    }
}
