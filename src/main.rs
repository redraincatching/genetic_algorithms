use std::thread;

mod travelling_salesperson;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let berlin = thread::spawn(|| {
        let _ = travelling_salesperson::analyse_dataset("./datasets/berlin52.tsp");
    });

    let kro = thread::spawn(|| {
        let _ = travelling_salesperson::analyse_dataset("./datasets/kroA100.tsp");
    });

    let pr = thread::spawn(|| {
        let _ = travelling_salesperson::analyse_dataset("./datasets/pr1002.tsp");
    });

    berlin.join().unwrap();
    kro.join().unwrap();
    pr.join().unwrap();

    Ok(())
}
