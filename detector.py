
import os
import sys
import cv2
import caffe
import numpy as np
from PIL import Image, ImageDraw
import skimage.color
import argparse


#IMG_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]
IMG_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

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
	bbs = np.zeros( (sh[0] / args.stride + 1, sh[1] / args.stride + 1, 4) )

	detect_height = caffenet.blobs[args.input].data.shape[2]
	detect_width = caffenet.blobs[args.input].data.shape[3]
	for y in xrange(0, sh[0], args.stride):
		for x in xrange(0, sh[1], args.stride):
			win = im[y:y+detect_height, x:x+detect_width,:]
			win = caffenet.transformer.preprocess(args.input, win)

			caffenet.blobs[args.input].data[...] = win
			caffenet.forward()
			heatmap[y / args.stride, x / args.stride] = caffenet.blobs[args.output_layer].data.flatten()[1]
			bbs[y / args.stride, x / args.stride] = caffenet.blobs[args.bb_layer].data.flatten()

	return heatmap, bbs


def get_output_name(f, out_dir, tag):
	base = os.path.basename(f)
	ext = base[-4:]
	base = base[:-4]

	sdir = os.path.join(out_dir, base)
	if not os.path.exists(sdir):
		os.mkdir(sdir)

	return os.path.join(sdir, "%s%s" % (tag, ext))


def draw_bb(draw, x, y, xEnd, yEnd, color='blue', thick=5):
	for t in xrange(thick):
		draw.rectangle( [x-t, y-t, xEnd + t, yEnd + t], outline=color)
			

def save_heatmap(heatmap, size, heat_out):
	heatmap = (heatmap / heatmap.max() * 255).astype(dtype=np.uint8)
	#heatmap = cv2.resize(heatmap, (im_np.shape[1], im_np.shape[0]), interpolation=cv2.INTER_NEAREST)
	heatmap = cv2.resize(heatmap, size, interpolation=cv2.INTER_NEAREST)
	print heatmap.shape
	expanded = np.zeros( (heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
	expanded[:,:,2] = heatmap
	cv2.imwrite(heat_out, expanded)
	

def handle_image(fn, caffenet, args):
	im_original = Image.open(fn)
	im_original = im_original.convert(mode='RGB')
	detect_height = caffenet.blobs[args.input].data.shape[2]
	detect_width = caffenet.blobs[args.input].data.shape[3]

	heatmaps = list()
	best_detection = (0, 0, 0, 0)
	for scale in IMG_SCALES:
		im = im_original.resize( (int(im_original.size[0] * float(scale)), int(im_original.size[1] * float(scale))), 
			resample=Image.BILINEAR)
		im_np = np.array(im.getdata()).reshape(im.size[1], im.size[0], 3)

		if args.hsv:
			im_np = skimage.color.rgb2hsv(im_np)
		elif args.float:
			im_np /= 255.

		heatmap, bbs = detect(im_np, caffenet, args)

		heat_out = get_output_name(fn, args.out_dir, "heatmap_%.2f" % scale)
		save_heatmap(heatmap, im_original.size, heat_out)
		heatmaps.append(heatmap)

		bb_out = get_output_name(fn, args.out_dir, "bb_%.2f" % scale)
		
		if args.bbr:
			y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
			bb = bbs[y, x, :]
			y = int( (y * args.stride + bb[0] * detect_height) / scale)
			x = int( (x * args.stride + bb[1] * detect_width) / scale) 
			yEnd = int(y + (min(bb[2], 1) * detect_height) / scale)
			xEnd = int(x + (min(bb[3], 1) * detect_width) / scale) 
		else:
			y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
			x = int(x * args.stride / scale)
			y = int(y * args.stride / scale)
			xEnd = int(x + detect_width / scale)
			yEnd = int(y + detect_height / scale)
		score = heatmap.max()
		#print scale, x, y, xEnd, yEnd, score
		if score > best_detection[0]:
			best_detection = (score, x, y, xEnd, yEnd, scale)

		bb_im = im_original.copy()

		draw = ImageDraw.Draw(bb_im)
		draw_bb(draw, x, y, xEnd, yEnd)
		bb_im.save(bb_out)

	bb_out = get_output_name(fn, args.out_dir, "bb_best_%.2f" % best_detection[-1])

	draw = ImageDraw.Draw(im_original)
	score, x, y, xEnd, yEnd, scale = best_detection
	draw_bb(draw, x, y, xEnd, yEnd)
	im_original.save(bb_out)


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
	parser.add_argument("--bb-layer", type=str, default="bb",
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
	parser.add_argument("-f", "--float", default=False, action="store_true",
				help="Scale read in images to [0-1]")
	parser.add_argument("--hsv", default=False, action="store_true",
				help="Convert images to HSV colorspace")
	parser.add_argument("--bbr", default=False, action="store_true",
				help="Do bounding box regression")
	
	return parser.parse_args()

	
	

if __name__ == "__main__":
	args = get_args()
	main(args)

