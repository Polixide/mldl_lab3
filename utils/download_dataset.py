import subprocess

# Scarica il dataset
subprocess.run(["wget", "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])

# Estrai i file
subprocess.run(["unzip", "tiny-imagenet-200.zip", "-d", "data/tiny_imagenet"])