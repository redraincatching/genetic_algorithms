use std::{error::Error, sync::Arc};

use genetic_algorithms::{epoch, FitnessOrder, Generation};
use travelling_salesman::{initialise_with_values, read_tsp_file, TSPath};

mod travelling_salesman;

fn main() -> Result<(), Box<dyn Error>> {
    // TODO FOR THIS ASSIGNMENT
    // time computation
    // try variations on crossover/mutation rates and population sizes (maybe find a way to make that dynamic)
    // also try 50/50 split, then 100/0, 0/100 for each operator
    // test on 3 sizes
    // plot for each (both best and average against optima)
    // write up

    let berlin_dataset = read_tsp_file("./datasets/berlin52.tsp").expect("no file found");
    let berlin_arc = Arc::new(berlin_dataset);
    
    let mut berlin: Generation<TSPath> = Generation::new(60);
    initialise_with_values(&mut berlin, berlin_arc.clone(), 0.05);

    let mut lowest_found = f64::MAX;
    let mut best_found = Vec::new();

    let mut generations: usize = 0;
    while generations < 3000 {
        epoch(&mut berlin, FitnessOrder::Min);
        generations += 1;
        println!("current average fitness: {}", berlin.get_average_fitness());

        if berlin.get_best_fitness() < lowest_found {
            lowest_found = berlin.get_best_fitness();
            best_found = berlin.get_best_solution().get_path();
        }
    }
    println!("best fitness: {}", lowest_found);
    println!("best solution: {:?}", best_found);

    Ok(())
}
