import os
import cv2
import numpy as np
import tensorflow as tf 
from test_code import network
from test_code import guided_filter
from tqdm import tqdm


# resize array of image based on maintaining the aspect ratio
# the colors are based on the surrounding pixels by choosing the values inbetween the pixels
def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image

def cartoonize(input_image, model_path, final_out):
    tf.reset_default_graph()

    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
   
    with tf.Session(config=config) as sess:
        print("Loading Model")
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Model Loaded")

        print("Processing Input")
        image = input_image
        image = np.array(image)
        image = resize_crop(image)
        batch_image = (image.astype(np.float32) / 127.5) - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        print("Running Model")
        output = sess.run(final_out, feed_dict={input_photo: batch_image})
        print("Processing Output")
        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output


    

if __name__ == '__main__':
    model_path = 'saved_models'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)
    

    