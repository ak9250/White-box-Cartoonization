import runway
import numpy as np
import tensorflow as tf
from test_code import network
import test_code.cartoonize as cart
from PIL import Image
from test_code import cartoonize as ct


@runway.setup(options={'checkpoint': runway.file(is_directory=True)})
def setup(opts):
    pass
    
@runway.command('translate', inputs={'image': runway.image}, outputs={'image': runway.image})
def translate(net, inputs):
    print("Starting")
    output = ct.cartoonize(inputs['image'], "test_code/saved_models")
    print("Done")
    return Image.fromarray(output)

if __name__ == '__main__':
    runway.run(port=8889)