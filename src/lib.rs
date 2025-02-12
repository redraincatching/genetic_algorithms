use std::cmp::Ordering;
use rayon::prelude::*;
use rand::{thread_rng, Rng};

/// # Genotype 
/// the encoded model for phenotypic characteristics of a solution
pub trait Genotype
where Self: Sized + Clone {
    /// generate offspring of two parents
    fn crossover(x: &Self, y: &Self) -> (Self, Self);
    /// randomised change
    fn mutation(&self) -> Self;
    /// generate a new randomised version of itself, for populating empty generation
    fn random() -> Self 
        {unimplemented!()}
    /// calculate the fitness of this solution
    fn fitness(&self) -> f64;
}

/// Each individual generation 
#[derive(Debug)]
pub struct Generation<T: Genotype + std::fmt::Debug> {
    population: Vec<T>,
    temp_population: Vec<T>,
    average_fitness: f64,
    population_size: usize
}

impl<T: Genotype + std::fmt::Debug> Generation<T> {
    pub fn new(size: usize) -> Self {
        Generation {
            population: Vec::new(),
            temp_population: Vec::new(),
            average_fitness: 0.0,
            population_size: size
        }
    }

    pub fn get_average_fitness(&self) -> f64 {
        self.average_fitness
    }

    pub fn get_best_solution(&mut self) -> T {
        self.population.sort_by(|a, b| {
            if a.fitness() > b.fitness() {
                Ordering::Less
            } else if a.fitness() < b.fitness() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        self.population.first().unwrap().clone()
    }

    pub fn get_best_fitness(&mut self) -> f64 {
        self.population.sort_by(|a, b| {
            if a.fitness() > b.fitness() {
                Ordering::Less
            } else if a.fitness() < b.fitness() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        self.population.first().unwrap().fitness()
    }

    pub fn get_population_size(&self) -> usize {
        self.population_size
    }

    pub fn push(&mut self, item: T) {
        self.population.push(item);
    }
}

/// initialise with random, unseeded population
pub fn initialise<T: Genotype + std::fmt::Debug>(gen: &mut Generation<T>) {
    for _ in 0..gen.population_size {
        gen.population.push(T::random());
    }
}

/// head-to-head tournament selection based on fitness
fn tournament_selection<T: Genotype>(solutions: &[T], order: &FitnessOrder) -> T {
    let mut rng = thread_rng(); 
    let s0_index = rng.gen_range(0..solutions.len());
    let s1_index = rng.gen_range(0..solutions.len());

    if 
        (*order == FitnessOrder::Max 
            && solutions.get(s0_index).unwrap().fitness() > solutions.get(s1_index).unwrap().fitness()) 
            || 
        (*order == FitnessOrder::Min 
            && solutions.get(s0_index).unwrap().fitness() < solutions.get(s1_index).unwrap().fitness()) {
        solutions.get(s0_index).unwrap().clone()
    } else {
        solutions.get(s1_index).unwrap().clone()
    }
}

// enum to determine how to determine whether we want max or min fitness
#[derive(PartialEq)]
pub enum FitnessOrder {Max, Min}

pub fn epoch<T: Genotype + std::fmt::Debug + Sync + Send>(gen: &mut Generation<T>, order: FitnessOrder) {
    let mut rng = thread_rng();

    // get average fitness of generation
    let mut fitness: f64 = gen.population.par_iter()
        .map(|solution| solution.fitness())
        .sum();
    fitness /= gen.population_size as f64;
    gen.average_fitness = fitness;

    if order == FitnessOrder::Max {
        // sort by fitness (in descending order)
        gen.population.par_sort_by(|a, b| {
            if a.fitness() > b.fitness() {
                Ordering::Less
            } else if a.fitness() < b.fitness() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
    } else {
        // sort by fitness (in ascending order)
        gen.population.par_sort_by(|a, b| {
            if a.fitness() < b.fitness() {
                Ordering::Less
            } else if a.fitness() > b.fitness() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
    }

    // keep best n solutions
    let best_n = 5;     // currently just keeping the top 2
    for i in 0..best_n {
        gen.temp_population.push(gen.population.get(i).unwrap().clone());
    }

    // set temp_pop from n to population_size with 2-element tournaments
    for _ in best_n..gen.population_size {
        gen.temp_population.push(tournament_selection(&gen.population, &order));
    }

    // clear out old population
    gen.population.clear();

    // stronger elitism - keep the best n solutions unchanged
    for i in 0..best_n {
        gen.population.push(gen.temp_population.get(i).unwrap().clone());
    }

    // perform crossover on all pairs without replacement
    for _ in best_n..gen.population_size / 2 {
        if let (Some(parent0), Some(parent1)) = (
            gen.temp_population.get(rng.gen_range(0..gen.population_size)),
            gen.temp_population.get(rng.gen_range(0..gen.population_size))
        ) {
            let (child0, child1) = Genotype::crossover(parent0, parent1);

            // perform mutations in this step as well
            gen.population.push(child0.mutation());
            gen.population.push(child1.mutation());
        }
    }

    // clear temp pop for next epoch
    gen.temp_population.clear();
}

