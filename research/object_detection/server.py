import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
sys.path.append("../")
sys.path.append("../..")
from object_detection.utils import ops as utils_ops

from utils import label_map_util

from utils import visualization_utils as vis_util
from PIL import Image
import grpc
import time
from concurrent import futures

import ObjectDetectionService_pb2
import ObjectDetectionService_pb2_grpc
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
print(categories)
category_index = label_map_util.create_category_index(categories)
def index_to_categories(indexs):
    cats = []
    for index in indexs:
        cats.append(category_index[index]['name'])
    return cats

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 7) ]

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name( tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

#def objectDetectTensor():
def objectDetectTensor(image_np):
    old = time.time()
    #image = Image.open("test_images/image1.jpg")
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    count = 0
    for scr in output_dict['detection_scores']:
        if scr > 0.5:
            count += 1
        else:
            break
    print(output_dict['detection_boxes'][:count])
    print(output_dict['detection_classes'][:count])
    print(output_dict['detection_scores'][:count])
    print((time.time() - old)*1000)
    names = index_to_categories(output_dict['detection_classes'][:count])
    x = [tup[0] for tup in output_dict['detection_boxes'][:count]]
    y = [tup[1] for tup in output_dict['detection_boxes'][:count]]
    z = [tup[2] for tup in output_dict['detection_boxes'][:count]]
    w = [tup[3] for tup in output_dict['detection_boxes'][:count]]
    return names, x, y, z, w

def getImage(queryImage):
    nparr = np.fromstring(queryImage.image, np.uint8)
    img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
    pil_im = Image.fromarray(img)
    rgbimg = Image.new("RGB", pil_im.size)
    rgbimg.paste(pil_im)
    cv2.imwrite('testtttttttt1.jpg',img)
    return load_image_into_numpy_array(rgbimg)   

class ObjectDetectionServiceServicer(ObjectDetectionService_pb2_grpc.ObjectDetectionServiceServicer):
    def objectDetect(self, request_iterator, context):
        for queryImage in request_iterator:
            #objectDetectTensor()
            names, xs, ys, zs, ws = objectDetectTensor(getImage(queryImage))
            yield  ObjectDetectionService_pb2.respondMessage(name=names, x=xs, y=ys, z=zs, w=ws)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ObjectDetectionService_pb2_grpc.add_ObjectDetectionServiceServicer_to_server(ObjectDetectionServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
