import numpy as np
import cv2
import face_recognition as fr
from glob import glob
import pickle
import utility
import random
import define_constants as const

print('-----------------------------------------------------\n')

with open('assets/pickles/n_people.pk', 'rb') as pickle_file:
    n_people_in_pickle = pickle.load(pickle_file)
print(f"Number of files that should be in '{const.PEOPLE_DIR}' directory : {n_people_in_pickle}")
people = glob(const.PEOPLE_DIR + '/*.*')
print(f"Number of files in '{const.PEOPLE_DIR}' directory : {len(people)}")

if n_people_in_pickle == len(people):

    names = list(map(utility.get_names, people))

    face_encode = np.load('assets/face_encodings/data.npy')

    print("\nInitiating camera...\n")
    cap = cv2.VideoCapture(const.n_camera)

    eye_blink_counter = 0
    eye_blink_total = 0
    random_blink_number = random.randint(const.n_min_eye_blink, const.n_max_eye_blink)
    frame_current_name = None

    while cap.isOpened():

        ret, frame = cap.read()

        frame_face_loc = fr.face_locations(frame)
        frame_face_landmarks = fr.face_landmarks(frame, frame_face_loc)
        frame_face_encode = fr.face_encodings(frame, frame_face_loc)

        for index, (loc, encode, landmark) in enumerate(zip(frame_face_loc, frame_face_encode, frame_face_landmarks)):

            score = fr.face_distance(face_encode, encode)
            index_match = np.argmin(score)

            if np.min(score) < const.face_recognition_threshold:

                temp_name = frame_current_name
                frame_current_name = names[index_match]
            else:
                frame_current_name = "Unknown"

            if not frame_current_name == "Unknown":

                left_eye_points = np.array(landmark['left_eye'], dtype=np.int32)
                right_eye_points = np.array(landmark['right_eye'], dtype=np.int32)

                EAR_avg = (utility.get_EAR_ratio(left_eye_points) + utility.get_EAR_ratio(right_eye_points)) / 2

                if EAR_avg < const.EAR_ratio_threshold:
                    eye_blink_counter += 1
                else:
                    if eye_blink_counter >= const.min_frames_eyes_closed:
                        eye_blink_total += 1

                    eye_blink_counter = 0

                if temp_name != frame_current_name:
                    eye_blink_total = 0
                    random_blink_number = random.randint(const.n_min_eye_blink, const.n_max_eye_blink)

                blink_message = f"Blink {random_blink_number} times, blinks:{eye_blink_total}"

                if utility.check_is_name_recorded(frame_current_name):

                    attendence_message = "Next Person"
                else:
                    attendence_message = " "
                face_box_color = const.default_face_box_color

                if random_blink_number == eye_blink_total:

                    if np.min(score) < const.face_recognition_threshold:
                        utility.record_attendence(frame_current_name)
                        face_box_color = const.success_face_box_color
                        random_blink_number = random.randint(const.n_min_eye_blink, const.n_max_eye_blink)
                        eye_blink_total = 0
                        eye_blink_counter = 0

                cv2.polylines(frame, [left_eye_points], True, const.eye_color, 1)
                cv2.polylines(frame, [right_eye_points], True, const.eye_color, 1)
                cv2.putText(frame, blink_message, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, const.text_in_frame_color, 2)
                cv2.putText(frame, attendence_message, (20, 450), cv2.FONT_HERSHEY_PLAIN, 1.5,
                            const.text_in_frame_color, 2)
            else:

                face_box_color = const.unknown_face_box_color

            cv2.rectangle(frame, (loc[3], loc[0]), (loc[1], loc[2]), face_box_color, 2)
            cv2.putText(frame, frame_current_name, (loc[3], loc[0] - 3), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        const.text_in_frame_color, 2)

        cv2.imshow("Webcam (Press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print(f"Run encode_faces.py to encode all faces in '{const.PEOPLE_DIR}' directory...")
