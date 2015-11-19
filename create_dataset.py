
import os
import sys
import argparse

SPRITE_OCCLUSION_HEIGHT = 0.75
MIN_OCCLUSION_BB_AREA = 0.1
MAX_OCCLUSION_BB_AREA = 1

IMAGE_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 2]
CROP_SIZE = (170, 80)


def getSprites(root, in_file, out_file, no_cache):
    pass


def main(args):
    sprites = getSprites(args.root, args.sprite_in_file, args.sprite_out_dir, args.sprite_no_cache)


def parse_args():
	parser = argparse.ArgumentParser(description="Creates a dataset")
	parser.add_argument("-r", "--root", default='.', 
				help="All filepaths are relative to the root")
	parser.add_argument("-s", "--sprite-in-file", default="original_sprites.txt", 
				help="File containing paths to each sprite")
	parser.add_argument("--sprite-out-dir", default="data/images/augmented_sprites",
				help="Output directory where the augmented sprites will be written")
	parser.add_argument("--sprite-no-cache", default=False, action="store_true",
				help="By default use any precomputed augmented sprites")

	parser.add_argument("-d", "--data-manifest", default='training_images.txt', 
				help="File containing paths to each training image")
	parser.add_argument("--crop-out-dir", default="data/images/random_crops",
				help="Output directory where the random crops are put")
	parser.add_argument("-n", "--num-crops", default=10000, type=int, 
				help="Number of random crops taken from each training image")
	parser.add_argument("--crops-no-cache", default=False, action="store_true",
				help="By default use previously computed random crops")

	parser.add_argument("-p", "--percent-positive", default=0.5, type=float, 
				help="Percentage of random crops that will be positive examples")
	parser.add_argument("-o", "--training-out-dir", default="data/images/training_data",
				help="Output directory where the random crops are put")
	parser.add_argument("-m", "--training-manifest", default="data/images/training_manifest.txt",
				help="Output Manifest File listing training images and labels")

	return parser.parse_args()
	

if __name__ == "__main__":
	args = parse_args()
	main(args)

