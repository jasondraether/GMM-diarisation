import os
from itertools import chain, combinations, product
from time import sleep

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1)))

data_directory = 'profile_data/'

file_powersets = []
if os.path.exists(data_directory):
    classes = os.listdir(data_directory)
    n_classes = len(classes)
    for class_id in range(n_classes):
        training_filenames = []
        class_directory = os.path.join(data_directory,classes[class_id])
        class_files = os.listdir(class_directory)
        for class_file in class_files:
            filename = os.path.join(class_directory,class_file)
            training_filenames.append([filename, class_id])

        print(training_filenames)

        training_powerset = powerset(training_filenames)
        training_powerset = list(filter(None,training_powerset))
        for t in training_powerset:
            print(t)
        file_powersets.append(training_powerset)

    powerset_product = list(product(*file_powersets))
    for x in powerset_product:
        for f in x:
            f[0][-1]
