import subprocess
import os
import shutil

# Download the dataset
subprocess.run(["wget", "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])

# Extract the files
subprocess.run(["unzip", "tiny-imagenet-200.zip", "-d", "dataset/tiny_imagenet"])

# Convert the data into the ImageFolder format of pytorch
with open('dataset/tiny_imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'dataset/tiny_imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'dataset/tiny_imagenet/tiny-imagenet-200/val/images/{fn}', f'dataset/tiny_imagenet/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('dataset/tiny_imagenet/tiny-imagenet-200/val/images')