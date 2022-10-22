from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from services.yolo.yolov3_tf2.models import YoloV3, YoloV3Tiny
from services.yolo.yolov3_tf2.utils import load_darknet_weights

flags.DEFINE_string('weights', 'weights/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', 'weights/yolov3.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


class Trainer(object):
    def weight_train(_argv):
        if FLAGS.tiny:
            yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            yolo = YoloV3(classes=FLAGS.num_classes)
        yolo.summary()
        logging.info('model created')

        load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
        logging.info('weights loaded')

        img = np.random.random((1, 320, 320, 3)).astype(np.float32)
        output = yolo(img)
        logging.info('sanity check passed')

        yolo.save_weights(FLAGS.output)
        logging.info('weights saved')

    def train():
        try:
            app.run(Trainer.weight_train)
        except SystemExit:
            pass

if __name__ == '__main__':
    Trainer.train()