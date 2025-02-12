/// # Operators Used
///
/// crossover operators:
/// - PMX
/// - OX
/// see comparative analysis of c.o. by KUMAR, KUMAR, KARAMBIR
///
/// mutation operators:
/// - RSM
/// - PSM
/// - SM
/// see mutation operators by ABDOUN, ABOUCHABAKA, TAJANI

use std::{collections::HashSet, error::Error, time::Instant, slice::Iter};
use std::sync::Arc;
use bimap::BiMap;
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use csv::Writer;
use tspf::{self, Tsp, TspBuilder};
use genetic_algorithms::{epoch, FitnessOrder, Generation, Genotype};

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub enum CrossoverOperator {
    PMX,
    OX,
    All
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub enum MutationOperator {
    RSM,
    PSM,
    SM,
    All
}

impl CrossoverOperator {
    pub fn iterator() -> Iter<'static, CrossoverOperator> {
        static OPERATORS: [CrossoverOperator; 3] = [CrossoverOperator::PMX, CrossoverOperator::OX, CrossoverOperator::All];
        OPERATORS.iter()
    }
}

impl MutationOperator {
    pub fn iterator() -> Iter<'static, MutationOperator> {
        static OPERATORS: [MutationOperator; 4] = [MutationOperator::RSM, MutationOperator::PSM, MutationOperator::SM, MutationOperator::All];
        OPERATORS.iter()
    }
}

#[derive(Debug, Clone)]
pub struct TSPath {
    data: Arc<Tsp>,
    path: Vec<usize>,
    mutation_rate: f64,
    crossover: CrossoverOperator,
    mutator: MutationOperator
}

impl TSPath {
    pub fn new(dataset: Arc<Tsp>, mutation_rate: f64, crossover: CrossoverOperator, mutator: MutationOperator) -> Self {
        let nodes = dataset.node_coords();
        let mut keys: Vec<usize> = nodes.keys().cloned().collect();

        // perform fisher-yates shuffle
        // hash key vector is randomised per run of the program
        // but not per instance of the ga
        keys.shuffle(&mut thread_rng());

        TSPath {
            data : dataset.clone(),
            path : keys,
            mutation_rate,
            crossover,
            mutator
        }
    }

    pub fn length(&self) -> usize {
        self.path.len()
    }

    pub fn get_path(&self) -> &Vec<usize> {
        &self.path
    }

    pub fn get_crossover(&self) -> &CrossoverOperator {
        &self.crossover
    }

    pub fn get_mutator(&self) -> &MutationOperator {
        &self.mutator
    }
}

impl Genotype for TSPath {
    fn crossover(x: &Self, y: &Self) -> (Self, Self) {
        let mut rng = thread_rng();

        // choose which crossover operator with equal chance of either
        match x.get_crossover() {
            CrossoverOperator::OX => {
                partially_mapped_crossover(x, y)
            },
            CrossoverOperator::PMX => {
                order_crossover(x, y)
            },
            _ => {
                if rng.gen_bool(0.5) {
                    partially_mapped_crossover(x, y)
                } else {
                    order_crossover(x, y)
                }
            }
        }
    }

    fn mutation(&self) -> Self {
        let mut rng = thread_rng();

        // check that mutation will occur
        if rng.gen::<f64>() < self.mutation_rate {
            // choose which mutation operation occurs

            match self.get_mutator() {
                MutationOperator::SM => {
                    return swap_mutation(self)
                },
                MutationOperator::RSM => {
                    return reverse_sequence_mutation(self)
                },
                MutationOperator::PSM => {
                    return partial_shuffle_mutation(self)
                },
                _ => {
                    // probabilities weighted in order of increasing destructiveness
                    let operator = rng.gen_range(1..=100);
                    if operator <= 25 {
                        return swap_mutation(self)
                    } else if operator <= 75 {
                        return reverse_sequence_mutation(self)
                    } else {
                        return partial_shuffle_mutation(self)
                    }

                }
            }

        }

        self.clone()
    }

    /// # fitness of solution
    /// represented as the total length of the round trip
    /// all cities are connected, and we use euclidean distances
    /// 
    /// ## known optimal distances for each dataset
    /// - berlin52: 7542
    /// - kroA100: 21282
    /// - pr1002: 259045
    /// see symmetric tsp
    #[allow(clippy::get_first)]
    fn fitness(&self) -> f64 {
        let mut total_distance = 0.0;
        let len = self.length();
        let map = self.data.node_coords();

        // get euclidiean distance between c and c + 1, wrapping back to start
        // \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
        for c in 0..len {
            let c_0 = self.path.get(c).unwrap();
            let c_1 = self.path.get((c + 1) % len).unwrap();

            let pos_c_0 = map.get(c_0).expect("city not found").pos();
            let pos_c_1 = map.get(c_1).expect("city not found").pos();
            
            total_distance +=
                ((pos_c_0[0] - pos_c_1[0]).powi(2) +
                 (pos_c_0[1] - pos_c_1[1]).powi(2)
                 ).sqrt();
        }

        total_distance
    }
}

pub fn read_tsp_file(filename: &str) -> Option<Tsp> {
    TspBuilder::parse_path(filename).ok()
}

/// initialise with predetermined dataset and values
pub fn initialise_with_values(gen: &mut Generation<TSPath>, dataset: Arc<Tsp>, mutation_rate: f64, crossover: CrossoverOperator, mutator: MutationOperator) {
    for _ in 0..gen.get_population_size() {
        gen.push(TSPath::new(dataset.clone(), mutation_rate, crossover.clone(), mutator.clone()));
    }
}

// --------------------
// Mutation Operators
// --------------------

/// # Partial Shuffle Mutation (PSM)
/// shuffle a section of the genes in the genotype
/// this one is far more destructive
fn partial_shuffle_mutation(parent: &TSPath) -> TSPath {
    let mut rng = thread_rng();
    let mut child = (*parent).clone();

    let i = rng.gen_range(0..parent.length() - 1);
    let j = rng.gen_range(i..parent.length());

    if i != j {
        let slice = &mut child.path[i..=j];
        slice.shuffle(&mut rng);
    }

    child
}

/// # Reverse Sequence Mutation (RSM)
/// we take a sequence between positions i and j, with i<j
/// the gene order of this sequence is then reversed
fn reverse_sequence_mutation(parent: &TSPath) -> TSPath {
    let mut rng = thread_rng();
    let mut child = (*parent).clone();

    let i = rng.gen_range(0..parent.length() - 1);
    let j = rng.gen_range(i..parent.length());

    if i != j {
        let slice = &mut child.path[i..=j];
        slice.reverse();
    }

    child
}

/// # Swap Mutation (SM)
/// does what it says on the tin
fn swap_mutation(parent: &TSPath) -> TSPath {
    let mut rng = thread_rng();
    let mut child = (*parent).clone();

    let i = rng.gen_range(0..parent.length());
    let j = rng.gen_range(0..parent.length());

    if i != j {
        child.path.swap(i, j);
    }

    child
}

// --------------------
// Mutation Operators
// --------------------

/// # Partially Mapped Crossover (PMX)
/// let parent_0 consisting of genotypes a_0...a_n and parent_1 with genotypes b_0...b_n
/// let empty genomes child_0 and child_1 of length n
/// we take two random numbers in the range [0, n], i and j, with i < j
/// elements a_i...a_j are placed in child_1 in the same locations, and b_i...b_j similarly in child_0
/// we then create a bijective mapping as follows
/// f(a_k) = b_k, and f(b_k) = a_k
/// we copy the remaining elements in parent_0 to child_1, passing them through this mapping, and do the same from parent_1 to child_0, ensuring that we have no invalid tours
fn partially_mapped_crossover(parent_0: &TSPath, parent_1: &TSPath) -> (TSPath, TSPath) {
    let mut rng = thread_rng(); 
    let length = parent_0.length();

    let i = rng.gen_range(0..length - 1);
    let j = rng.gen_range(i+1..length);

    let mut child_0 = parent_0.clone();
    let mut child_1 = parent_1.clone();

    let mut mapping = BiMap::new();

    // copy middle sections from parents
    // and create mapping between elements
    for idx in i..=j {
        child_1.path[idx] = parent_0.path[idx];
        child_0.path[idx] = parent_1.path[idx];

        mapping.insert(parent_0.path[idx], parent_1.path[idx]);
    }

    // copy left and right sections across, through the mapping
    for idx in 0..length {
        if idx < i || idx > j {
            // parent_0 and child_1
            // check if the valuee is in the mapping
            let value = if mapping.contains_left(&parent_0.path[idx]) { 
                // if so, return the result of the map
                mapping.get_by_left(&parent_0.path[idx]).unwrap()
            } else { 
                // otherwise just return the value
                &parent_0.path[idx]
            };
            child_1.path[idx] = *value;

            // parent_1 and child_0
            // check if the valuee is in the mapping
            let value = if mapping.contains_right(&parent_1.path[idx]) { 
                // if so, return the result of the map
                mapping.get_by_right(&parent_1.path[idx]).unwrap()
            } else { 
                // otherwise just return the value
                &parent_1.path[idx]
            };
            child_0.path[idx] = *value;
        }
    }

    (child_0, child_1)
}

/// # Order Crossover (OX)
/// Select k random positions from parent_0 and copy them into child_0
/// iterate through parent_1 and copy each currently unused index into the next empty space in child_0
/// repeat this process using the n-k positions not chosen from parent_0
fn order_crossover(parent_0: &TSPath, parent_1: &TSPath) -> (TSPath, TSPath) {
    let mut rng = thread_rng();
    let length = parent_0.length();
    let k = rng.gen_range(0..length);

    let mut indices = HashSet::new();

    let mut child_0 = parent_0.clone();
    let mut child_1 = parent_1.clone();

    // set values in child arrays to usize::MAX to represent "empty" indices
    // it's unlikely that we ever need to represent that many cities, otherwise i'd use an option
    for idx in 0..length {
        child_0.path[idx] = usize::MAX;
        child_1.path[idx] = usize::MAX;
    }

    // generate k random positions
    while indices.len() < k {
        indices.insert(rng.gen_range(0..length));
    }

    // place the k selected values into child_0, and the other n-k values into child_1
    for idx in 0..length {
        if indices.contains(&idx) {
            child_0.path[idx] = parent_0.path[idx];
        } else {
            child_1.path[idx] = parent_0.path[idx];
        }
    }

    // fill out the empty spaces
    let mut child_0_idx = 0;
    let mut child_1_idx = 0;
    for idx in 0..length {
        // insert the next unused index
        if !child_0.path.contains(&parent_1.path[idx]) {
            // find the next empty space
            while child_0.path[child_0_idx] != usize::MAX {
                child_0_idx += 1;
            }
            child_0.path[child_0_idx] = parent_1.path[idx];
            child_0_idx += 1;
        }
        // repeat for child_1
        // this could possibly be an else instead of another if but i'm not 100% sure
        if !child_1.path.contains(&parent_1.path[idx]) {
            // find the next empty space
            while child_1.path[child_1_idx] != usize::MAX {
                child_1_idx += 1;
            }
            child_1.path[child_1_idx] = parent_1.path[idx];
            child_1_idx += 1;
        }
    }

    (child_0, child_1)
}

pub fn analyse_dataset(filepath: &str) -> Result<(), Box<dyn Error>> {
    let dataset = read_tsp_file(filepath).expect("no file found");
    let dataset_arc = Arc::new(dataset);

    // set up csv writer
    let filename = filepath.strip_prefix("./datasets/").unwrap();
    let mut writer = Writer::from_path(format!("output/{}", filename)).unwrap();
    // [IDENTIFIER, X, Y, Y_alternative]
    writer.write_record(["mutation_rate", "epoch", "best_fitness", "average_fitness"])?;

    let start = Instant::now();
    
    for crossover in CrossoverOperator::iterator() {
        for mutator in MutationOperator::iterator() {
            for step in (1..=7).step_by(1) {
                let mutation_rate = f64::from(step) * 0.01;

                let mut generations: usize = 0;
                let mut lowest_found = f64::MAX;
                let mut best_found = Vec::new();

                let mut city: Generation<TSPath> = Generation::new(200);
                initialise_with_values(&mut city, dataset_arc.clone(), mutation_rate, crossover.clone(), mutator.clone());
                
                let mut gen_since_improvement: usize = 0;

                // check for convergence, and also cap it because i'm on a laptop
                while gen_since_improvement < 300 && generations < 4000 {
                    epoch(&mut city, FitnessOrder::Min);
                    generations += 1;
                    gen_since_improvement += 1;
                    //println!("gen {}, since improvement: {}", generations, gen_since_improvement);

                    if city.get_best_fitness() < lowest_found {
                        lowest_found = city.get_best_fitness();
                        best_found = (*city.get_best_solution().get_path()).clone();

                        gen_since_improvement = 0;
                    }

                    writer.write_record([mutation_rate.to_string(), generations.to_string(), city.get_best_fitness().to_string(), city.get_average_fitness().to_string()])?;
                }
                writer.flush()?;

                println!("dataset: {} with crossover: {:?}, mutator: {:?}, and mutation rate: {}\nbest fitness: {}\nbest solution: {:?}", filename, crossover, mutator, mutation_rate, lowest_found, best_found);
            }

        }
    }

    let elapsed = start.elapsed();
    println!("time taken for {}: {:.2?}", filename, elapsed);

    Ok(())
}
    // TODO: write a plotting function and run it afterwards
    // also write something to plot the best path found 
