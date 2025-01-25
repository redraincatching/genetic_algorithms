import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# check if the script received a csv file argument
if len(sys.argv) < 2:
    print("please provide the csv file name as an argument.")
    sys.exit(1)

# get the csv file name from the arguments
csv_file = sys.argv[1]

# read the csv file into a pandas dataframe
df = pd.read_csv(csv_file)

# plotting the data
plt.plot(df['epoch'], df['average fitness'], label='average fitness', color='cyan')

# adding labels and title
plt.xlabel('epoch')
plt.ylabel('average fitness')
plt.title('average fitness over epochs')

# save as image with the same name as the problem
base_name, _ = os.path.splitext(csv_file)   # remove the extension from the file name
png_filename = f"{base_name}.png"           # add the .png extension
plt.savefig(png_filename)

# display the plot
plt.legend()
plt.show()
