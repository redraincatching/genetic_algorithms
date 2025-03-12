import sys
import os
import ast
import matplotlib.pyplot as plt

def parse_node_data(file_path):
    nodes = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_index = None
        
        for i, line in enumerate(lines):
            if line.startswith("NODE_COORD_SECTION"):
                start_index = i + 1
                break
        
        if start_index is None:
            raise ValueError("NODE_COORD_SECTION not found in the file")
        
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodes[node_id] = (x, y)
    
    return nodes

def plot_tour(nodes, indices):
    tour_x = []
    tour_y = []

    for i in indices:
        x, y = nodes[i]
        tour_x.append(x)
        tour_y.append(y)

    tour_x.append(tour_x[0])
    tour_y.append(tour_y[0])

    plt.figure(figsize=(8, 6))
    plt.plot(tour_x, tour_y, 'o-', color='cyan', markerfacecolor='navy', markersize=4, label="tour path")
    plt.title('node tour')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.grid(True)
    # plt.show()

def main():
    if len(sys.argv) < 3:
        print("usage: python plot_tsp_path.py <tsp_file> <indices>")
        sys.exit(1)
    
    node_file = sys.argv[1]
    try:
        indices = ast.literal_eval(sys.argv[2])
        if not isinstance(indices, list):
            raise ValueError("The input should be a list of indices.")
    except (ValueError, SyntaxError):
        print("Error: Invalid list format. Please ensure the indices are passed as a valid Python list.")
        sys.exit(1)
    
    nodes = parse_node_data(node_file)
    
    plot_tour(nodes, indices)

    base_name, _ = os.path.splitext(os.path.basename(node_file))  # remove the extension from the file name
    png_filename = f"{base_name}_tour.png"  
    plt.savefig(os.path.join("output", png_filename))

if __name__ == "__main__":
    main()

