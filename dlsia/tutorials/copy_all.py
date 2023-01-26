"""
Save this file in an empty directory and call it "copy_all.py".
Then type:
python copy_all.py

Now you have all notebooks in an empty directory. Start a fresh jupyter session
or run "./run_all.sh <my_kernel>" to execute all notebooks (this will take some time)
"""
import os
import shutil
import stat
from dlsia import tutorials

path = tutorials.__file__[:-11]
for file in os.listdir(path):
    if file[-5:] == "ipynb":
        print(path + file)
        this_file = path + file
        shutil.copy(this_file, ".")
shutil.copy(path + "run_all.sh", ".")
# make executable
os.chmod("./run_all.sh", stat.S_IRWXU)
