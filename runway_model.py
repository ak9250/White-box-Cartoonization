import runway
import numpy as np
import tensorflow as tf
from test_code import network
import test_code.cartoonize as cart
from PIL import Image
from test_code import cartoonize as ct


# tag for the function that is called when you get started
@runway.setup(options={'checkpoint': runway.file(is_directory=True)})
def setup(opts):
    pass 
    
# translate is the function that is called when you input a image, specify the input and output types
@runway.command('translate', inputs={'image': runway.image}, outputs={'image': runway.image})
def translate(net, inputs):
    output = ct.cartoonize(inputs['image'], "test_code/saved_models")
    return Image.fromarray(output)


if __name__ == '__main__':
    runway.run(port=8889)