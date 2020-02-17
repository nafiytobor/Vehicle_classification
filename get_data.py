# Create the identification hash for your submission 

import hashlib
import glob
submission_hash = hashlib.md5().hexdigest()
is_hash = glob.glob("./hash.txt")
if len(is_hash)==0:
    with open("./hash.txt", "w") as f:
        f.write(submission_hash)


!wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
!wget http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
!wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat

!mkdir data
!tar -xvzf ./cars_train.tgz -C ./data/
!rm -r ./cars_train.tgz
!tar -xvzf ./car_devkit.tgz -C ./data/
!rm -r ./car_devkit.tgz
print("Finished downloading files")


from utils import create_dataset
create_dataset()