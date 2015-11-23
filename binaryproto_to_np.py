
import caffe.proto.caffe_pb2
import numpy as np
import sys

protobuf_mean_file = sys.argv[1]
np_out_file = sys.argv[2]

bp = caffe.proto.caffe_pb2.BlobProto()
bp.ParseFromString(open(protobuf_mean_file).read())

channels = bp.channels if bp.HasField('channels') else bp.shape.dim[1]
height = bp.height if bp.HasField('height') else bp.shape.dim[2]
width = bp.width if bp.HasField('width') else bp.shape.dim[3]

arr = np.zeros( (channels, height, width) )

for c in xrange(channels):
	for h in xrange(height):
		for w in xrange(width):
			arr[c,h,w] = bp.data[(c * height + h) * width + w]


np.save(np_out_file, arr)



