import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# check if the script received a csv file argument
if len(sys.argv) < 3:
    print("please provide the csv file name and fitness type as an argument")
    sys.exit(1)

# get the csv file name from the arguments
csv_file = sys.argv[1]
fitness_type = sys.argv[2]

# check that the fitness type is valid
if fitness_type not in ['best_fitness', 'average_fitness']:
    print("invalid fitness type. please choose 'best_fitness' or 'average_fitness'")
    sys.exit(1)

# read the csv file into a pandas dataframe
df = pd.read_csv(csv_file)

# validate the file
required_columns = ['crossover_rate', 'mutation_rate', 'epoch', 'best_fitness', 'average_fitness']
for col in required_columns:
    if col not in df.columns:
        print(f"missing column: {col}")
        sys.exit(1)

# plotting the data
for crossover_rate in df['crossover_rate'].unique():
    crossover_data = df[df['crossover_rate'] == crossover_rate]
    
    for mutation_rate in crossover_data['mutation_rate'].unique():
        mutation_data = crossover_data[crossover_data['mutation_rate'] == mutation_rate]

        plt.plot(mutation_data['epoch'], mutation_data[fitness_type], label=f'crossover {crossover_rate}, mutation {mutation_rate}')

plt.xlabel('epoch')
plt.ylabel(f'{fitness_type.replace("_", " ").title()}')
plt.title(f'{fitness_type.replace("_", " ").title()} over epochs')

# add legend
# plt.legend(title="crossover and mutation rate", loc='upper right')

# save as image with the same name as the problem
base_name, _ = os.path.splitext(csv_file)   # remove the extension from the file name
png_filename = f"{base_name}_{fitness_type}.png"  # add the fitness type to the filename
plt.savefig(png_filename)

# display the plot
# plt.show()
