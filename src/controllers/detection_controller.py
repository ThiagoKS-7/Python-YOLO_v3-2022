import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import os
import base64
from src.parameters import get_YOLO_img_to_base64_response_params as get_Yolo

classes_path, weights_path, tiny, img_size, num_classes, output_path = get_Yolo()

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print("weights loaded")

class_names = [c.strip() for c in open(classes_path).readlines()]
print("classes loaded")


class YOLO_img_to_base64_response(object):
    def predict(image):
        image_name = 'meme.jpg' # image.filename
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(open(image_name, "rb").read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, img_size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print("time: {}".format(t2 - t1))

        print("detections:")
        for i in range(nums[0]):
            print(
                "\t{}, {}, {}".format(
                    class_names[int(classes[0][i])],
                    np.array(scores[0][i]),
                    np.array(boxes[0][i]),
                )
            )
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(output_path + "detection.jpg", img)
        print("output saved to: {}".format(output_path + "detection.jpg"))

        _, img_encoded = cv2.imencode(".png", img)
        response = img_encoded.tostring()

        os.remove(image_name)
        return base64.b64encode(response)