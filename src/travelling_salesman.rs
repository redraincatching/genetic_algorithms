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
/// see mutation operators by ABDOUN, ABOUCHABAKA, TAJANI

// TODOS
// add concurrency
// precalculate distance matrix
// maybe change mutation_rate or distribution dynamically during the process? maybe based on a fitness delta

use std::collections::HashSet;
use std::sync::Arc;
use bimap::BiMap;
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use tspf::{self, Tsp, TspBuilder};
use genetic_algorithms::{Genotype, Generation};

#[derive(Debug, Clone)]
pub struct TSPath {
    data: Arc<Tsp>,
    path: Vec<usize>,
    mutation_rate: f64
}

impl TSPath {
    pub fn new(dataset: Arc<Tsp>, mutation_rate: f64) -> Self {
        let nodes = dataset.node_coords();
        let mut keys: Vec<usize> = nodes.keys().cloned().collect();

        // perform fisher-yates shuffle
        // hash key vector is randomised per run of the program
        // but not per instance of the ga
        keys.shuffle(&mut thread_rng());

        TSPath {
            data : dataset.clone(),
            path : keys,
            mutation_rate
        }
    }

    pub fn length(&self) -> usize {
        self.path.len()
    }

    pub fn get_path(&self) -> Vec<usize> {
        self.path.clone()
    }
}

impl Genotype for TSPath {
    fn crossover(x: &Self, y: &Self) -> (Self, Self) {
        let mut rng = thread_rng();

        // choose which crossover operator with equal chance of either
        if rng.gen_bool(0.5) {
            partially_mapped_crossover(x, y)
        } else {
            order_crossover(x, y)
        }
    }

    fn mutation(&self) -> Self {
        let mut rng = thread_rng();

        // check that mutation will occur
        if rng.gen::<f64>() < self.mutation_rate {
            // choose which mutation operator with equal chance of either
            if rng.gen_bool(0.05) {
                 // this one is far more destructive so we want to hit it less often
                return partial_shuffle_mutation(self)
            } else {
                return reverse_sequence_mutation(self)
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
pub fn initialise_with_values(gen: &mut Generation<TSPath>, dataset: Arc<Tsp>, mutation_rate: f64) {
    for _ in 0..gen.get_population_size() {
        gen.push(TSPath::new(dataset.clone(), mutation_rate));
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

    for i in 0..parent.length() {
        let j = rng.gen_range(i..parent.length());
        child.path.swap(i, j);
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
