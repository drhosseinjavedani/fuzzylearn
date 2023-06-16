from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


def feature_improtance(args, **kwargs):
    """Feature improtance"""
    features = kwargs["feature"]
    y_for_color_code = kwargs["y_for_color_code"]
    lhss = kwargs["lhss"]
    plt.figure(figsize=(6, 8))

    # Get the number of columns in the array
    num_columns = lhss.shape[1]

    # Generate the x-axis values for the bars
    x_values = np.arange(num_columns)
    x_values = features
    # Example list of numbers to determine colors
    colors = list(set(y_for_color_code.iloc[:, 0].to_list()))
    colors = [float(x) for x in colors]

    # Create a color map
    cmap = plt.cm.get_cmap("viridis")  # Choose a colormap, such as 'cool'

    # Create lines connecting the bars
    for i in range(lhss.shape[0] - 1):
        plt.plot(
            lhss[i, :],
            x_values,
            linestyle="-",
            color=cmap(colors[int(y_for_color_code.iloc[i, :].to_list()[0])]),
            label=str(int(y_for_color_code.iloc[i, :].to_list()[0])),
        )

    plt.legend(y_for_color_code)

    # Set labels and a title for the plot
    plt.xlabel("Levels")
    plt.ylabel("Features/Variables")
    plt.title("Fuzzified Features Map")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))

    # Display a legend for the rows
    plt.legend(by_label.values(), by_label.keys())

    # Show the plot
    plt.show()
