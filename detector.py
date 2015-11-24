
import os
import sys
import cv2
import caffe
import numpy as np
from PIL import Image, ImageDraw
import argparse


IMG_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]

def init_caffe(args):
	if args.gpu >= 0:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	caffenet = caffe.Net(args.caffe_model, args.caffe_weights, caffe.TEST)
	transformer = caffe.io.Transformer({args.input: caffenet.blobs[args.input].data.shape}, resize=True)
	transformer.set_transpose(args.input, (2,0,1))
	if args.mean_file:
		transformer.set_mean(args.input, np.load(args.mean_file).mean(1).mean(1))
	else:
		transformer.set_mean(args.input, np.asarray([-1 * args.shift] * 3))
	transformer.set_raw_scale(args.input, args.scale)

	transformer.set_channel_swap(args.input, (2,1,0))
	caffenet.transformer = transformer

	return caffenet


def get_im_files(args):
	img_root = args.img_dir
	im_files = list()
	for line in open(args.manifest, 'r').readlines():
		line = line.rstrip()
		im_files.append(os.path.join(img_root, line))

	return im_files


def detect(im, caffenet, args):
	sh = im.shape
	heatmap = np.zeros( (sh[0] / args.stride + 1, sh[1] / args.stride + 1) )
	bbs = []

	detect_height = caffenet.blobs[args.input].data.shape[2]
	detect_width = caffenet.blobs[args.input].data.shape[3]
	for y in xrange(0, sh[0], args.stride):
		for x in xrange(0, sh[1], args.stride):
			win = im[y:y+detect_height, x:x+detect_width,:]
			win = caffenet.transformer.preprocess(args.input, win)

			caffenet.blobs[args.input].data[...] = win
			caffenet.forward()
			heatmap[y / args.stride, x / args.stride] = caffenet.blobs[args.output_layer].data.flatten()[1]

	return heatmap, bbs


def get_output_name(f, out_dir, tag):
	base = os.path.basename(f)
	ext = base[-4:]
	base = base[:-4]

	return os.path.join(out_dir, "%s_%s%s" % (base, tag, ext))


def draw_bb(draw, x, y, width, height, color='blue', thick=5):
	for t in xrange(thick):
		draw.rectangle( [x-t, y-t, x + width + t, y + height + t], outline=color)
			

def save_heatmap(heatmap, size, heat_out):
	heatmap = (heatmap / heatmap.max() * 255).astype(dtype=np.uint8)
	#heatmap = cv2.resize(heatmap, (im_np.shape[1], im_np.shape[0]), interpolation=cv2.INTER_NEAREST)
	heatmap = cv2.resize(heatmap, size, interpolation=cv2.INTER_NEAREST)
	cv2.imwrite(heat_out, heatmap)
	

def handle_image(fn, caffenet, args):
	im_original = Image.open(fn)
	im_original = im_original.convert(mode='RGB')
	detect_height = caffenet.blobs[args.input].data.shape[2]
	detect_width = caffenet.blobs[args.input].data.shape[3]

	heatmaps = list()
	for scale in IMG_SCALES:
		im = im_original.resize( (int(im_original.size[0] * float(scale)), int(im_original.size[1] * float(scale))), 
			resample=Image.BILINEAR)
		im_np = np.array(im.getdata()).reshape(im.size[1], im.size[0], 3)

		heatmap, bbs = detect(im_np, caffenet, args)

		heat_out = get_output_name(fn, args.out_dir, "heatmap_%.2f" % scale)
		save_heatmap(heatmap, im_original.size, heat_out)
		heatmaps.append(heatmap)

		bb_out = get_output_name(fn, args.out_dir, "bb_%.2f" % scale)
		
		y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
		print scale, x, y
		x = x * args.stride
		y = y * args.stride

		draw = ImageDraw.Draw(im)
		draw_bb(draw, x, y, detect_width, detect_height)
		im.save(bb_out)


def main(args):
	caffenet = init_caffe(args)
	im_files = get_im_files(args)

	try:
		os.makedirs(args.out_dir)
	except:
		pass

	for x, f in enumerate(im_files):
		try:
			print f
			handle_image(f, caffenet, args)

		except:
			print "Exception occured processing %s" % f
			raise


def get_args():
	parser = argparse.ArgumentParser(description="Detection with Caffe")
	parser.add_argument("caffe_model", 
				help="The model definition file (e.g. deploy.prototxt)")
	parser.add_argument("caffe_weights", 
				help="The model weight file (e.g. net.caffemodel)")
	parser.add_argument("manifest", 
				help="File listing image paths relative to img_dir.  Determines order of features")
	parser.add_argument("img_dir", 
				help="Root directory for the images")
	parser.add_argument("out_dir", 
				help="Root directory for the output images")

	
	parser.add_argument("--stride", type=int, default=15,
				help="Stride of the sliding window")
	parser.add_argument("--output-layer", type=str, default="probs",
				help="Name of the output layer")

	parser.add_argument("-m", "--mean-file", type=str, default="",
				help="Optional mean file for input normalization")
	parser.add_argument("--gpu", type=int, default=-1,
				help="GPU to use for running the network")
	parser.add_argument("-a", "--scale", type=float, default=1.0,
				help="Optional scale factor")
	parser.add_argument("-b", "--shift", type=float, default=0.0,
				help="Optional shift factor")
	parser.add_argument("-i", "--input", type=str, default="data",
				help="Name of input blob")
	
	return parser.parse_args()

	
	

if __name__ == "__main__":
	args = get_args()
	main(args)

