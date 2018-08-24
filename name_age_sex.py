import os
import time

import cv2
import imutils
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dlibmodel.face_dlib import dlib_API
from imutils.video import FileVideoStream
import random
import tensorflow as tf
import inception_resnet_v1
from imutils.face_utils import FaceAligner

SKIP_FRAMES = 5


class dlibFace:
    # Initialize some variables
    def __init__(self):
        self.img_size = 160
        self.known_face_names = []
        self.known_face_encodings = []
        self.net = cv2.dnn.readNetFromCaffe('../model/deploy.prototxt.txt',
                                            caffeModel='../model/res10_300x300_ssd_iter_140000.caffemodel')
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.font = ImageFont.truetype('msyh.ttc', 16, encoding='utf-8')
        self.path = 'video_train'
        self.threshold = 0.65
        self.encodingfile = 'known_face_encodings'
        self.r_model = dlib_API()
        self.sess, self.age, self.gender, self.train_mode, self.images_pl = self.load_network('./tfmodel/')

    def train(self):
        if os.path.exists('%s.npy' % self.encodingfile):
            self.known_face_encodings = np.load('%s.npy' % self.encodingfile)
            for child_dir in os.listdir(self.path):
                self.known_face_names.append(child_dir)
        else:
            self.facecoding()

    def facecoding(self):
        def endwith(s, *endstring):
            resultArray = map(s.endswith, endstring)
            if True in resultArray:
                return True
            else:
                return False

        for child_dir in os.listdir(self.path):
            child_path = os.path.join(self.path, child_dir)
            known_face_encodings_each = []
            for dir_image in os.listdir(child_path):
                if endwith(dir_image, 'jpg'):
                    image = cv2.imdecode(np.fromfile(os.path.join(child_path, dir_image), dtype=np.uint8),
                                         cv2.IMREAD_UNCHANGED)
                    # load the input image and construct an input blob for the image
                    # by resizing to a fixed 300x300 pixels and then normalizing it
                    # image = cv2.imread(args["image"])
                    (h, w) = image.shape[:2]
                    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    # image = imutils.resize(image, width=300)

                    # pass the blob through the network and obtain the detections and predictions
                    # print("[INFO] computing object detections...")
                    self.net.setInput(blob)
                    detections = self.net.forward()
                    # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                    detections = detections[detections[:, :, :, 2] >= self.threshold]
                    # loop over the detections
                    for i in range(0, detections.shape[0]):
                        # extract the confidence (i.e., probability) associated with the prediction
                        # confidence = detections[i, 2]

                        # compute the (x, y)-coordinates of the bounding box for the object
                        box = detections[i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        location = [(int(startY), int(endX), int(endY), int(startX))]
                        known_face_encodings_each.append(
                            self.r_model.face_encodings(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), location, 1)[0])
            self.known_face_names.append(child_dir)
            self.known_face_encodings.append(np.mean(known_face_encodings_each, axis=0))
        np.save(self.encodingfile, self.known_face_encodings)

    @staticmethod
    def load_network(model_path):
        sess = tf.Session()

        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
        train_mode = tf.placeholder(tf.bool)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        # gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        gender = tf.nn.softmax(gender_logits)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore model!")
        else:
            pass
        return sess, age, gender, train_mode, images_pl

    def video(self, input, output):
        fa = FaceAligner(self.r_model.pose_predictor_68_point, desiredFaceWidth=self.img_size)
        # Get a reference to webcam #0 (the default one)
        # video_capture = cv2.VideoCapture(0)
        # video_capture = VideoStream(src=0).start()
        # video_capture = FileVideoStream(input).start()
        video_capture = FileVideoStream(
            'rtsp://admin:adminVFIZVY@172.16.3.67:554/Streaming/Channels/101?transportmode=multicast').start()
        time.sleep(1)

        out = None
        count = 0
        while True:
            # Grab a single frame of video
            frame = video_capture.read()
            # frame = imutils.resize(frame, width=720)
            (h, w) = frame.shape[:2]
            # (delta_h, delta_w) = (int(0.2 * h), int(0.3 * w))
            # frame = frame[delta_h:h - delta_h, delta_w:w - delta_w - 300]
            # (h, w) = frame.shape[:2]
            # if out is None:
            #     out = cv2.VideoWriter(output, self.fourcc, 11, (w, h), True)

            count += 1
            if count % SKIP_FRAMES != 0:
                # cv2.imshow('Video', frame)
                continue

            input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.r_model.face_detector(input_img, 1)  # 增加上采样可增加图片分辨率，降低处理速度
            faces = np.empty((len(detections), self.img_size, self.img_size, 3))
            for i, d in enumerate(detections):
                startX, startY, endX, endY, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                # extract the confidence (i.e., probability) associated with the prediction
                # confidence = detections[i, 2]

                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                # if confidence < 0.5:
                #     continue

                # compute the (x, y)-coordinates of the bounding box for the object
                # start1 = time.time()
                raw_landmark_set = self.r_model.pose_predictor_68_point(frame, d)
                # end1 = time.time()
                face_encoding = np.array(
                    self.r_model.face_encoder.compute_face_descriptor(frame, raw_landmark_set))
                # print('encoding time %d\t%d' % ((end1 - start1) * 1000, (time.time() - end1) * 1000))
                #
                # # start2 = time.time()
                # # See if the face is a match for the known face(s)
                results = self.r_model.face_distance(self.known_face_encodings, face_encoding)  # Euclidean distance
                # # end2 = time.time()
                # # print('1NN time %d\t%d-1' % (((end2 - start2) * 1000), len(self.known_face_encodings)))
                #
                # # If a match was found in known_face_encodings, just use the first one.
                match_index = np.argmin(results)
                if results[match_index] > 0.5:
                    continue
                name = self.known_face_names[int(match_index)]
                # print(name, results[match_index])

                # tf
                faces[i, :, :, :] = fa.align(input_img, gray, detections[i])
                ages, genders = self.sess.run([self.age, self.gender], feed_dict={
                    self.images_pl: faces, self.train_mode: False})
                # print(genders)
                age, gender = int(ages[i]), "女" if genders[i][0] > 0.5 else "男"

                # if abs(genders[i][0] - genders[i][1]) < 0.3:
                #     print('年龄%d,性别%s' % (age, '未知'))
                # else:
                #     print('年龄%d,性别%s' % (age, gender))

                y = d.top() - 20 if d.top() - 20 > 20 else d.top() + 20
                cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame)
                draw.text((d.left(), y), '%s,%d,%s' % (name, age, gender), font=self.font, fill=(255, 0, 0))
                # draw.text((d.left(), y), '%d,%s' % (age, gender), font=self.font, fill=(255, 0, 0))
                frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

            # out.write(frame)
            # cv2.imshow('Video2', face_frame)
            # Display & save the resulting frame
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q') or not video_capture.more():
                break

        # Release handle to the webcam
        cv2.destroyAllWindows()
        video_capture.stop()
        # out.release()


if __name__ == '__main__':
    model = dlibFace()
    video_path = r'video_test.mov'
    out_path = 'ag.avi'
    # model.train()
    model.video(video_path, out_path)
