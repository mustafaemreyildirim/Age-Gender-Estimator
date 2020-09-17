'''
usage:
    train.py  --utkroot=<str>   

'''
import os
import glob
from docopt import docopt
import json
finlist =[]
def process(root):
    
    for i, path in enumerate(glob.glob(os.path.abspath('{}/*.jpg'.format(root)))):
        
        age, gender = os.path.basename(path).split('_')[:2]
        age, gender = int(age), int(gender)
        if age > 0 and age < 100 and (gender == 0 or gender == 1):
            if gender==0:gender=1
            else:gender=0
            lastlist = {"path":path,"age":age,"gender":gender} 
            finlist.append(lastlist)

args = docopt(__doc__)
process(args['--utkroot'])

