import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/home/adminuser/deep-learning/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_alexnet/deploy.prototxt',caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel',caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
batch_size=50
net.blobs['data'].reshape(batch_size,3,227,227)
image_dir='../data/images/'
from os import listdir
from os.path import isfile, join
import time
image_files=[]
for f in listdir(image_dir):
	if isfile(join(image_dir,f)):
		image_files.append(f)
start=0
end=len(image_files)
f=open("../data/imgfeat.tsv","w")
e=open("../data/imgfeat_err.tsv","w")
ps=time.clock()
ec=0
while True:
	if start>=end:
		break
	print('preprocessing...')
	s=time.clock()
	for i in range(batch_size):
		cur_idx=start+i
		if cur_idx>=end:
			cur_idx=start
		cur_img=image_files[cur_idx]
		try:
			net.blobs['data'].data[i]=transformer.preprocess('data', caffe.io.load_image(image_dir+cur_img))
		except:
			e.write(cur_img+'\n')
			net.blobs['data'].data[i]=transformer.preprocess('data', caffe.io.load_image(image_dir+image_files[0]))
			ec=ec+1
			print('Error count = '+str(ec))
	print("Done in "+str((time.clock()-s)/60)+" minutes.")	
	print('testing...')
	s=time.clock()
	out=net.forward()
	print("Done in "+str((time.clock()-s)/60)+" minutes.")
	data=net.blobs['fc7'].data
	for i in range(batch_size):
		cur_idx=start+i
		if cur_idx<end:
			cur_img=image_files[start+i]
			f.write(cur_img)
			for col in range(4096):
				f.write("\t"+str(data[i][col]))
			f.write("\n")
	start+=batch_size
f.close()
e.close()
print("Done in "+str((time.clock()-ps)/60)+" minutes.")
print("Errors : "+str(ec))