use rand::{thread_rng, Rng};

use genetic_algorithms::Genotype;

#[derive(Clone, Debug)]
pub struct OneMax(Vec<u8>);

impl Genotype for OneMax {
    fn crossover(x: &Self, y: &Self) -> (Self, Self) {
        let mut rng = thread_rng();

        // choose swapping index
        let index = rng.gen_range(0..30);

        // get slices of parents
        let x_slice_0 = &x.0[0..index];
        let x_slice_1 = &x.0[index..30];
        let y_slice_0 = &y.0[0..index];
        let y_slice_1 = &y.0[index..30];

        // collect into vectors
        let child_0: Vec<u8> = x_slice_0.iter().chain(y_slice_1.iter()).copied().collect();
        let child_1: Vec<u8> = y_slice_0.iter().chain(x_slice_1.iter()).copied().collect();

        (OneMax(child_0), OneMax(child_1))
    }        

    fn mutation(&self) -> Self {
        let mut rng = thread_rng();
        let mut next = self.clone();

        // chance of mutation
        if rng.gen::<f64>() < 0.01 {
            // which digit will be mutated
            let idx = rng.gen_range(0..30);
            next.0[idx] ^= 1;
        }

        next
    }

    /// literally just the number of 1s in the string
    fn fitness(&self) -> f64 {
        self.0.iter().filter(|d| **d == 1 ).count() as f64
    }

    /// generates a bitstring of length 30, with each bit randomly assigned 0 or 1
    fn random() -> Self {
        let mut rng = thread_rng();
        OneMax((0..30)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { 0 })
            .collect())
    }
}
