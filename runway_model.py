import runway
import numpy as np
import tensorflow as tf
from test_code import network
import test_code.cartoonize
from PIL import Image

g = tf.get_default_graph()
sess = tf.InteractiveSession(graph=g)

@runway.setup(options={'checkpoint': runway.file(is_directory=True)})
def setup(opts):
    net = network()
    return net
    
@runway.command('translate', inputs={'image': runway.image}, outputs={'image': runway.image})
def translate(net, inputs):
    original_size = inputs['image'].size
    img = inputs['image'].resize((256, 256))
    output = net.generate(img)
    output = np.clip(output, -1, 1)
    output = ((output + 1.0) * 255 / 2.0)
    output = output.astype(np.uint8)
    return Image.fromarray(output).resize(original_size)

if __name__ == '__main__':
    runway.run(port=8889)