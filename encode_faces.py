import numpy as np
from glob import glob
import pickle
from tqdm import tqdm
import face_recognition as fr
import utility
import define_constants as const

print('-----------------------------------------------------\n')


people = glob(const.PEOPLE_DIR + '/*.*')
print(f"Number of files in '{const.PEOPLE_DIR}' directory : {len(people)}")


images = list(map(utility.get_images, people))


face_encode = []
is_face_found = True
print('Encoding faces...')
for index, img in enumerate(tqdm(images)):
    try:
        face_encode.append(fr.face_encodings(img, num_jitters=const.n_face_encoding_jitters)[0])
    except Exception as e:
        print(f"Face not found in file : {people[index]}, replace it.")
        is_face_found = False
        break

if is_face_found:
    print('Encoding completed...')

    np.save('assets/face_encodings/data.npy', face_encode)
    print(f"Data saved for {index+1} images...")


    with open('assets/pickles/n_people.pk', 'wb') as pickle_file:
        pickle.dump(len(people), pickle_file)
    print('Pickle file saved...')
