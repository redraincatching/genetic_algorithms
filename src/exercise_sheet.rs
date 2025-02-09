use csv::Writer;
use std::{error::Error, process::Command};

use genetic_algorithms::{epoch, initialise, FitnessOrder, Generation};
use one_max::OneMax;
use target_string::TargetString;
use deceptive_landscape::DeceptiveString;

mod one_max;
mod target_string;
mod deceptive_landscape;

fn main() -> Result<(), Box<dyn Error>> {
    // python environment
    let python_path = ".venv/bin/python3";
    let plotting_script = "plotting/plot_fitness.py";
    // set up csv writer
    let mut writer = Writer::from_path("output/one_max.csv").unwrap();
    writer.write_record(["epoch", "average fitness"])?;
    let mut idx = 0;

    // one max problem
    let mut one_max_problem: Generation<OneMax> = Generation::new(30);
    initialise(&mut one_max_problem);

    // we know here that the max fitness must be 30
    while one_max_problem.get_best_fitness() < 30.0 {
        epoch(&mut one_max_problem, FitnessOrder::Max);
        writer.write_record([idx.to_string(), one_max_problem.get_average_fitness().to_string()])?;
        idx += 1;
    }
    writer.flush()?;

    // plot graph
    let output = Command::new(python_path)
        .arg(plotting_script)
        .arg("output/one_max.csv")
        .output()?;

    if !output.status.success() {
        eprintln!("error: {}", String::from_utf8_lossy(&output.stderr));
    }

    println!("--- one max problem ---");
    println!("best solution:\n{:?}\nfitness: {}", one_max_problem.get_best_solution(), one_max_problem.get_best_fitness());

    // reset the writer
    writer = Writer::from_path("output/target_string.csv").unwrap();
    writer.write_record(["epoch", "average fitness"])?;
    idx = 0;

    // search for target string
    let mut target_string: Generation<TargetString> = Generation::new(30);
    initialise(&mut target_string);

    // we know here that the max fitness must be 30
    while target_string.get_best_fitness() < 30.0 {
        epoch(&mut target_string, FitnessOrder::Max);
        writer.write_record([idx.to_string(), target_string.get_average_fitness().to_string()])?;
        idx += 1;
    }
    writer.flush()?;

    // plot graph
    Command::new(python_path)
        .arg(plotting_script)
        .arg("output/target_string.csv")
        .output()?;
    
    println!("--- target string ---");
    println!("target string: 101011010111010111111101010000");
    println!("best solution:\n{:?}\nfitness: {}", target_string.get_best_solution(), target_string.get_best_fitness());

    // reset the writer
    writer = Writer::from_path("output/deceptive_string.csv").unwrap();
    writer.write_record(["epoch", "average fitness"])?;
    idx = 0;

    // search for target string
    let mut deceptive_string: Generation<DeceptiveString> = Generation::new(30);
    initialise(&mut deceptive_string);

    // if we get greater or equal to 30 we've hit either the good solution or the best
    while deceptive_string.get_best_fitness() < 30.0 {
        epoch(&mut deceptive_string, FitnessOrder::Max);
        writer.write_record([idx.to_string(), deceptive_string.get_average_fitness().to_string()])?;
        idx += 1;
    }
    writer.flush()?;

    // plot graph
    Command::new(python_path)
        .arg(plotting_script)
        .arg("output/deceptive_string.csv")
        .output()?;
    
    println!("--- deceptive string ---");
    println!("target string: 101011010111010111111101010000");
    println!("best solution:\n{:?}\nfitness: {}", deceptive_string.get_best_solution(), deceptive_string.get_best_fitness());

    Ok(())
}
