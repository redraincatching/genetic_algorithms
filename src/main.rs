use std::thread;

mod travelling_salesperson;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO FOR THIS ASSIGNMENT
    // time computation
    // try variations on crossover/mutation rates and population sizes (maybe find a way to make that dynamic)
    // also try 50/50 split, then 100/0, 0/100 for each operator
    // test on 3 sizes
    // plot for each (both best and average against optima)
    // write up

    let berlin = thread::spawn(|| {
        let _ = travelling_salesperson::analyse_dataset("./datasets/berlin52.tsp");
    });

    //let kro = thread::spawn(|| {
    //    let _ = travelling_salesperson::analyse_dataset("./datasets/kroA100.tsp");
    //});
    //
    //let pr = thread::spawn(|| {
    //    let _ = travelling_salesperson::analyse_dataset("./datasets/pr1002.tsp");
    //});

    berlin.join().unwrap();
    //kro.join().unwrap();
    //pr.join().unwrap();

    Ok(())
}
