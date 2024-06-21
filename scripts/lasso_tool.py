import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.colors as mcolors
import itertools
import matplotlib.cm as cm

def onselect(verts):
    global color_cycle  # Declare color_cycle as global
    path = Path(verts)
    ind = np.nonzero(path.contains_points(xy))[0]
    try:
        new_color = next(color_cycle)
    except StopIteration:
        color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS)  # Restart color cycle if exhausted
        new_color = next(color_cycle)
    color_dict[new_color] = ind  # Store indices under the new color
    color_history.append((new_color, ind))  # Keep track of the order of selections
    update_scatter_colors()

def update_scatter_colors():
    # Initialize the colors array with a default blue color for all points
    colors = np.full((len(xy), 4), [0, 0, 1, 0.6])  # Default color blue with alpha 0.6
    for color, indices in color_dict.items():
        # Ensure each color is converted to an RGBA format and assigned correctly
        colors[indices] = mcolors.to_rgba(color)
    scatter.set_facecolor(colors)  # Update the face colors of the scatter plot
    fig.canvas.draw_idle()  # Redraw the figure to update the display

def on_key(event):
    if event.key == 'backspace' and color_history:
        last_color, last_indices = color_history.pop()  # Remove the last selection
        for color, indices in list(color_dict.items()):
            if np.array_equal(indices, last_indices):
                del color_dict[color]  # Remove the last color from the dictionary
                break
        update_scatter_colors()

def on_close(event):
    global data  # Ensure data is accessible in this function
    # Create a new labels array based on the lasso selections
    new_labels = np.array(data['hdbscan_labels'])  # Start with a copy of the original labels
    for color, indices in color_dict.items():
        new_labels[indices] = np.unique(new_labels[indices])[0]  # Assign a unique label to each selected group

    # Save the modified labels back to the npz file
    np.savez(output_file_path, **data, hdbscan_labels=new_labels)
    print("Modified labels saved to:", output_file_path)

def lasso_tool(file_path, output_file_path):
    global data  # Declare data as global to access it in on_close
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    embedding = data["embedding_outputs"]

    global fig, ax, scatter, xy, color_dict, color_cycle, color_history
    fig, ax = plt.subplots()
    xy = embedding  # Assuming embedding is an Nx2 numpy array
    color_dict = {}
    color_history = []  # To track the history of selections
    scatter = ax.scatter(xy[:, 0], xy[:, 1], s=30, facecolor='blue', edgecolor='none', alpha=0.6)

    # Generate 100 unique colors
    color_cycle = itertools.cycle([cm.tab20(i/20) for i in range(20)] + [cm.tab20b(i/20) for i in range(20)] + [cm.tab20c(i/20) for i in range(20)] + [cm.Paired(i/12) for i in range(12)] + [cm.Pastel1(i/9) for i in range(9)] + [cm.Pastel2(i/8) for i in range(8)])
    lasso = LassoSelector(ax, onselect)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)  # Connect the close event to the on_close function

    plt.show()
