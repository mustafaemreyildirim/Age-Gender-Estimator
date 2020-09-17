'''
usage:
    train.py  --file=<str>  --file2=<str>   [options]

options:
    --root=<str>        utkfile root [default: crop_part1]
'''
import json
import os
import glob
from scipy.io import loadmat
import numpy as np
import codecs, json 
from datetime import datetime
from docopt import docopt


args = docopt(__doc__)
def calc_age(taken, dob):
        birth = datetime.fromordinal(max(int(dob) - 366, 1))
        if birth.month < 7:
            return taken - birth.year
        else:
            return taken - birth.year - 1

def ds(typee):

    path = '/home/memre/Desktop/face_gender_project/data_operation/mat_files/'+typee+'.mat'
    dataset = loadmat(path)
    image_path_array = dataset[typee]['full_path'][0, 0][0]
    lastlist =[]
    #dobs
    dob = dataset[typee]['dob'][0, 0][0]
    #photo taken
    photo_taken = dataset[typee]['photo_taken'][0, 0][0]
    #gender and age arrays
    gender_arr = dataset[typee]['gender'][0, 0][0]
    age_arr = np.array([calc_age(photo_taken[i], dob[i]) for i in range(len(dob))])
    valid_age_range = np.isin(age_arr, [x for x in range(101)])

    #facepoints compare
    face_score_threshold = 1

    face_score = dataset[typee]['face_score'][0, 0][0]
    second_face_score = dataset[typee]['second_face_score'][0, 0][0]   
            
    face_score_mask = face_score > face_score_threshold
    second_face_score_mask = np.isnan(second_face_score)
    unknown_gender_mask = np.logical_not(np.isnan(gender_arr))

    mask = np.logical_and(face_score_mask, second_face_score_mask)
    mask = np.logical_and(mask, unknown_gender_mask)
    mask = np.logical_and(mask, valid_age_range)
            
    image_path_array = image_path_array[mask].tolist()
    strtoadd = typee+'_crop/'
    image_path_array = [x + string for x in image_path_array]              
    gender_arr = gender_arr[mask].tolist()
    age_arr = age_arr[mask].tolist()
    list1 = [x[0] for x in image_path_array]

    lastlist = [{"path":list1[i],"age":age_arr[i],"gender":gender_arr[i]} for i in range(len(list1))]
    
    return lastlist


def process(root):
    
    finlist =[]
    for i, path in enumerate(glob.glob(os.path.abspath('/home/memre/Desktop/face_gender_project/data_operation/{}/*.jpg'.format(root)))):
        
        age, gender = os.path.basename(path).split('_')[:2]
        age, gender = int(age), int(gender)
        if age > 0 and age < 100 and (gender == 0 or gender == 1):
            if gender==0:gender=1
            else:gender=0
            path = root+'/'+path
            lastlist = {"path":path,"age":age,"gender":gender} 
            finlist.append(lastlist)
    
    return finlist

first = ds(args['--file'])
second = ds(args['--file2'])
utk = process(args['--root'])
a = first + second + utk
with open('data.json', 'w') as outfile:
    json.dump(a, outfile)