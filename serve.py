import os
import moxel.space

from MNISTTester import MNISTTester

####################
# directory settings
script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = script_dir + '/mnist/data/'
model_path = script_dir + '/models/mnist-cnn'

#####################################
# prediction test with MNIST test set
mnist = MNISTTester(
            model_path=model_path,
            data_path=data_path)


def predict(img):
    img_bw = img.to_PIL().convert('L')
    img_bw = moxel.space.Image.from_PIL(img_bw)
    out = mnist.predict(img_bw.to_stream())
    print(out)
    return {'out': out}


