import os
import shutil

# list of file endings to remove
file_endings = ['.egg-info', '__pycache__', '.tox', '.vscode']

# folders to recursively delete
del_folders = []

# get list of folders to delete
for root, d_names, f_names in os.walk('.'):
    for dir in d_names:
        directory = root + '\\' + dir
        for file_ending in file_endings:
            if directory.endswith(file_ending):
                del_folders.append(root + '\\' + dir)

print(del_folders)

# delete folders
input('Are you sure (press enter to continue!)')

for folder in del_folders:
    shutil.rmtree(folder)
