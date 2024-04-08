#!/bin/bash
# This script reads a path to a folder containing the images and generates a gif video
# For creating the gif, a pallette is created to improve the quality of the video
# The output video is saved in the same folder as the images
# The images are assumed to be in the format: 00001.png, 00002.png, etc.

# Ask for the path to the folder containing the images
# echo "Enter the path to the folder containing the images: "
# read path
path="/home/david/research/robotsimplesim/results"
video_name="video"

# In that path there will be folders with different experiment names. 
# For each folder there will be run names.
# Inside each run folder there will be a folder there will be the images

# Get all the experiment names
experiment_names=$(ls $path)

# For each experiment name
for experiment_name in $experiment_names
do
    echo "Generating videos in folder: "
    echo "$path/$experiment_name/"
    # Get all the run names (only get the folders as there are also files in the folder)
    run_paths=$(find $path/$experiment_name/ -maxdepth 1 -type d)

    # For each run name
    for run_path in $run_paths
    do
        echo "Generating videos for run: "
        echo "$run_path/"
        # The framerate of the video is set to 10 fps
        # The images are assumed to be in the format: 00001.png, 00002.png, etc.

        # Get the whole path to the images which is the path, the experiment name and the run name
        image_path=$(echo $run_path/)

        # Create a palette
        ffmpeg -y -f image2 -framerate 2 -i $image_path%04d.png -vf palettegen palette.png

        # Create the video
        ffmpeg -y -f image2 -framerate 2 -i $image_path%04d.png -i palette.png -filter_complex "fps=10,paletteuse" $image_path/$video_name.gif

        # Remove the palette
        rm $image_path/palette.png
    done
done
