import os
import shutil

def group_videos_by_serial(src_folder):
    # Set the destination folder to be the same as the source folder
    dest_folder = src_folder
    # Create a dictionary to store the serial numbers of the videos and their parts
    videos = {}
    # Iterate through the files in the source folder
    for file in os.listdir(src_folder):
        if len(file) > 8: # do not examine analized, combined videos
        # Get the serial number and part number from the file name
            serial = file[4:8]
            part = file[2:4]
            # If this serial number is not already in the dictionary, add it
            if serial not in videos:
                videos[serial] = []
            # Add the file to the list of parts for this serial number
            if file.endswith('.MP4'):
                videos[serial].append(file)
    
    # Create a list of the serial numbers, sorted in ascending order
    serial_numbers = sorted(videos.keys())
    # Loop through the serial numbers and move the corresponding videos to the appropriate folder
    if len(serial_numbers) > 1:
        for i, serial in enumerate(serial_numbers):
            # Determine the destination folder based on the index of the serial number
            if i == 0:
                dest = 'Naive'
            elif i == 1:
                dest = 'Inj'
            elif i == 2:
                dest = 'Dummy'
            else:
                dest = f'Extra{i - 2}'
            
            # Create the destination folder if it doesn't already exist
            os.makedirs(os.path.join(dest_folder, dest), exist_ok=True)
            # Move each part of the video to the destination folder
            for part in videos[serial]:
                shutil.move(os.path.join(src_folder, part), os.path.join(dest_folder, dest))
    

def group_all_videos_by_serial(root_folder):
    # Iterate through the subfolders of the root folder and their subfolders recursively
    leaf_folders = get_all_leaf_folders(root_folder)
    for leaf in leaf_folders:
        files = os.listdir(leaf)
        if any(file.endswith('.MP4') for file in files):
            # Create a set to store the serial numbers of the videos
            serial_numbers = set()
            # Iterate through the files in the folder
            for file in files:
                # Get the serial number from the file name
                serial = file[4:8]
                # Add the serial number to the set
                serial_numbers.add(serial)
            # If the set has more than one element, the folder contains videos with different serial numbers
            if len(serial_numbers) > 1:
                group_videos_by_serial(leaf)


def get_all_leaf_folders(root_folder):
    leaf_folders = []
    # Iterate through the subfolders of the root folder
    for root, dirs, __ in os.walk(root_folder):
        # If the current folder has no subfolders, add its path to the list
        if not dirs:
            leaf_folders.append(root)
    return leaf_folders
    
if __name__ == '__main__':
    # folder = input('input folder path:')
    folder = 'D:\\Data\\HO27'
    group_all_videos_by_serial(folder)