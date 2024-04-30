# Import Python Package
import pandas as pd
import numpy as np
import seaborn as sns
import os
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

def visualize2d(root_path):
    # Initialize empty lists to store the information
    sizes = []
    resolutions = []
    color_distributions = []

    # Iterate over each image file in each subdirectory
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Load the image file using OpenCV
                img_path = os.path.join(dirpath, filename)
                img = cv2.imread(img_path)

                # Extract the size of the image
                size = os.path.getsize(img_path)
                print(size)
                sizes.append(size)

                # Extract the resolution of the image
                resolution = img.shape[:2]
                print("Image Shape: ",img.shape)
                print("Resolution: ",resolution)
                resolutions.append(resolution)

                # Extract the color distribution of the image
                color_distribution = np.bincount(img.flatten(), minlength=256)
                print(color_distribution)
                color_distributions.append(color_distribution)

    # Convert the lists to numpy arrays for easier manipulation
    sizes = np.array(sizes)
    resolutions = np.array(resolutions)
    color_distributions = np.array(color_distributions)

    #Plot a histogram of the image sizes
    plt.hist(sizes)
    plt.title("Distribution of Image Sizes")
    plt.xlabel("File Size (bytes)")
    plt.ylabel("Number of Images")
    plt.show()

    # Create a scatter plot figure with plotly
    fig = px.scatter(x=resolutions[:, 0], y=resolutions[:, 1], title="Distribution of Image Resolutions")

    # Customize the plot
    fig.update_layout(
        xaxis_title="Width (pixels)",
        yaxis_title="Height (pixels)",
        showlegend=False,
        hovermode="closest",
        width=800,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )

    # Show the plot
    fig.show()
    visualize3d(resolutions)

def visualize3d(resolutions):
    # Create a dataframe with the resolutions
    df = pd.DataFrame(resolutions, columns=['width', 'height'])

    # Create a 3D scatter plot with plotly
    fig = px.scatter_3d(df, x='width', y='height', z=df.index,
                        title='Distribution of Image Resolutions',
                        labels={'width': 'Width (pixels)',
                                'height': 'Height (pixels)',
                                'index': 'Image Index'},
                        color=df.index)

    # Customize the plot
    fig.update_traces(marker=dict(size=2, line=dict(width=0.5)))

    # Show the plot
    fig.show()

#root_path = "../../DATA298-FinalProject/Removed_Duplicates"
#visualize2d(root_path)
