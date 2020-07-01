import runway
import numpy as np
import tensorflow as tf
from test_code import network
import test_code.cartoonize as cart
from PIL import Image
from test_code import cartoonize as ct

g = tf.get_default_graph()
sess = tf.InteractiveSession(graph=g)

@runway.setup(options={'checkpoint': runway.file(is_directory=True)})
def setup(opts):
    # net = cart.cartoonize(args.checkpoint)
    # return net
    pass
    
@runway.command('translate', inputs={'image': runway.image}, outputs={'image': runway.image})
def translate(net, inputs):
    output = ct.cartoonize(inputs['image'], "test_code/saved_models")
    output = output.astype(np.uint8)
    return Image.fromarray(output)

if __name__ == '__main__':
    runway.run(port=8889)