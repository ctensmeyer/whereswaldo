
import os
import sys
import glob
import lmdb
import random
import shutil
import argparse
import warnings
import numpy as np

import caffe.proto.caffe_pb2
import matplotlib.pyplot as pl
from skimage import io, transform, filters, color

SPRITE_OCCLUSION_HEIGHT = 0.75
SPRITE_ROTATE = (-0.5235987755982988, 0.5235987755982988) # +- 30 in radians
SPRITE_SCALE = (0.75, 1.25)
SPRITE_SHEAR = (-0.3490658503988659, 0.3490658503988659) #+- 20 in radians

IMAGE_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 2]
CROP_SIZE = (170, 170)

HSV_SATURATION_SIGMA_RANGE = (0.1,1.4)

QUIET = False

LMDBS = dict()

def init_dbs(args):
    for method in ["train", "val"]:
        for data in ["images", "bbs"]:
            fn = "%s_%s_%s_lmdb" % (args.lmdb_prefix, method, data)
            if os.path.exists(fn):
                try:
                    shutil.rmtree(fn)
                except:
                    pass
            LMDBS["%s_%s" % (method, data)] = open_db(fn)


def close_dbs():
    for env, txn in LMDBS.values():
        txn.commit()
        env.sync()
        env.close()

def open_db(db_file):
    env = lmdb.open(db_file, readonly=False, map_size=int(2 ** 42), writemap=True)
    txn = env.begin(write=True)
    return env, txn, 0

def get_handles(method, data):
    env, txn, next_key = LMDBS["%s_%s" % (method, data)]
    LMDBS["%s_%s" % (method, data)] = (env, txn, next_key + 1)
    return env, txn, next_key
    

def append_db(im, label, method, bb=None):
    env_im, txn_im, next_key_im = get_handles(method, "images")

    datum = caffe.proto.caffe_pb2.Datum()

    datum.label = label
    datum.channels = im.shape[2]
    datum.height = im.shape[0]
    datum.width = im.shape[1]
    if (im.dtype == np.float):
        datum.float_data = datum_im.transpose(2, 0, 1).tostring()
    else:
        datum.data = datum_im.transpose(2, 0, 1).tostring()

    txn_im.put(next_key_im, datum.SerializeToString())

    if bb is not None:
        env_bb, txn_bb, next_key_bb = get_handles(method, "bbs")
        assert next_key_bb == next_key_im

        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 4
        datum.height = 1
        datum.width = 1 
        datum.float_data = bb.tostring()

        txn_bb.put(next_key_bb, datum.SerializeToString())
        

    
#Read in file of filenames
def loadFileNames(txt_file, root="."):
    names = []
    with open(txt_file, 'r') as f:
        for line in f:
            names.append(os.path.join(root, line.strip()))

    return names

def randomBeta(low, high, beta,alpha):
    return (random.betavariate(alpha,beta)* (high-low)) + low

def otsu_alpha(im):
    threshold = filters.threshold_otsu(im[:,:,3])
    im[im[:,:,3] < threshold, 3] = 0
    im[im[:,:,3] >= threshold, 3] = 255

    return im


def createSprites(num_sprites, root, in_file, out_folder):
    filenames = loadFileNames(in_file, root)
    
    augmented = []

    print "Generating %d sprites:" % (num_sprites)

    out = out_folder + "/"
    for i in xrange(num_sprites):
        fn = random.choice(filenames)
        sprite = io.imread(fn)
        sh = sprite.shape

        #Fix alpha channel
        sprite = otsu_alph(sprite)

        #Drop alpha box on spite. 
        startY = randomBeta(sh[0]*SPRITE_OCCLUSION_HEIGHT, sh[0], 5,1)
        startX = randomBeta(0, sh[1], 5,1) #More likely to start square on left side of image.

        endY = randomBeta(startY, sh[0], 0.5,5)
        endX = randomBeta(startX, sh[1], 0.5, 5)

        sprite[startY:endY, startX:endX, 3] = 0
        
        imgLoc = out + "sprite_%d.png" % (i)

        io.imsave(imgLoc, sprite)
        augmented.append(imgLoc)
          
    return augmented


def getImages(num_imgs, root, in_file, out_dir, no_cache, create, name="images"):
    try:
        os.makedirs(out_dir)
    except:
        pass
    images = glob.glob(out_dir+"/*.png")

    if no_cache or len(images) < num_imgs:
        images = create(num_imgs, root, in_file, out_dir)       
    else:
        print "Sampling %d %s out of %d cached." % (num_imgs, name ,len(images))
        images = random.sample(images, num_imgs)

    return images

def createCrops(num_crops, root, in_file, out_folder):
    filenames = loadFileNames(in_file, root)
    
    crops = []
    images = []

    for fn in filenames:
        original_im = io.imread(fn)
        for scale in IMAGE_SCALES:
            resized_im = transform.rescale(original_im, scale)
            images.append(resized_im)

    if not QUIET:
        print "Generating %d crops:" % (num_crops)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        out = out_folder + "/"
        for i in xrange(num_crops):
            if i % 100 == 0 and not QUIET:
                print "Processed: %d" % (i)
            img = random.choice(images)
            sh = img.shape

            x = random.randrange(0, sh[1] - CROP_SIZE[1])
            y = random.randrange(0, sh[0] - CROP_SIZE[0])

            xEnd = x + CROP_SIZE[1]
            yEnd = y + CROP_SIZE[0]
        
            #print "Y: %d:%d, X: %d:%d" %(x, xEnd, y, yEnd)

            cr = img[y:yEnd, x:xEnd, :3]

            #print cr.shape

            imgLoc = out + "crop_%d.png" % (i)
            #io.imshow(cr)
            #io.show()
            io.imsave(imgLoc, cr)

            crops.append(imgLoc)
          
    return crops


def insert(sprite_im, crop_im, x, y, xEnd, yEnd):
    subcrop = crop_im[y:yEnd, x:xEnd, :]
    assert subcrop.shape[:2] == sprite_im.shape[:2]
    assert subcrop.shape[2] == 3
    assert sprite_im.shape[2] == 4
    mask = np.expand_dims(sprite_im[:,:,3] > 1, axis=2)


    subcrop = mask * sprite_im[:,:,:3] + (1 - mask) * subcrop
    crop_im[y:yEnd, x:xEnd, :] = subcrop[:,:,:]
    return crop_im


def make_instances(sprite_files, crop_files, root, out_dir, percent_positive):
    positive_dir = os.path.join(out_dir, "waldos")
    negative_dir = os.path.join(out_dir, "background")
    try:
        os.makedirs(positive_dir)
    except:
        pass
    try:
        os.makedirs(negative_dir)
    except:
        pass
    if not QUIET:
        print "Creating Dataset in %s" % out_dir

    sprites = map(lambda fn: io.imread(fn), sprite_files)

    positive_files = []
    negative_files = []
    for crop_fn in crop_files:
        is_positive = random.random() < percent_positive

        if is_positive:
            crop_im = io.imread(crop_fn)

            sprite_im = random.choice(sprites)

            #print sprite_im.shape

            #Randomly tranform sprite_im
            scale = (random.uniform(*SPRITE_SCALE), random.uniform(*SPRITE_SCALE))
            rotation = random.uniform(*SPRITE_ROTATE)
            shear = random.uniform(*SPRITE_SHEAR)

            trans = transform.AffineTransform(scale=scale, rotation=rotation, shear=shear)
            sprite_im = otsu_alpha(transform.warp(sprite_im, trans, mode="constant", cval=0, preserve_range=True))

            sprite_sh = sprite_im.shape
            #io.imshow(sprite_im[:,:,3])
            #io.show()
            #print sprite_im.shape, crop_im.shape
            #print

    
            x = random.randrange(0, CROP_SIZE[1] - sprite_sh[1])
            y = random.randrange(0, CROP_SIZE[0] - sprite_sh[0])

            xEnd = x + sprite_sh[1]
            yEnd = y + sprite_sh[0]

            #print "Y: %d:%d X: %d:%d" % (y,yEnd, x,xEnd)

            # insert sprite into crop
            inserted = insert(sprite_im, crop_im, x, y, xEnd, yEnd)
            #io.imshow(inserted)
            #io.show()

            hsv = color.rgb2hsv(inserted)
            sigma = random.uniform(*HSV_SATURATION_SIGMA_RANGE)
            hsv[:,:,1] = filters.gaussian_filter(hsv[:,:,1], sigma)

            img = color.hsv2rgb(hsv) #Unsure how to save HSV image

            imgLoc = os.path.join(positive_dir, "waldo_%d.png" % len(positive_files))
            io.imsave(imgLoc, img)
            positive_files.append(imgLoc)
        else:
            imgLoc = os.path.join(negative_dir, "background_%d.png" % len(positive_files))
            shutil.copyfile(crop_fn, imgLoc)
            negative_files.append(imgLoc)


    return positive_files, negative_files

def create_manifest(filename, positives, negatives):
    if not QUIET:
        print "Writing manifest to %s" % filename
    with open(filename, 'w') as out:
        for fn in positives:
            out.write("%s %d\n" % (fn, 1))
        for fn in negatives:
            out.write("%s %d\n" % (fn, 0))
    
def main(args):
    init_dbs(args)

    sprites = getImages(args.num_sprites, args.root, args.sprite_in_file, args.sprite_out_dir, args.sprites_no_cache, createSprites, "sprites")
    crops = getImages(args.num_crops, args.root, args.data_manifest, args.crop_out_dir, args.crops_no_cache, createCrops, "crops")
    positives, negatives = make_instances(sprites, crops, args.root, args.training_out_dir, args.percent_positive)

    # not strictly needed anymore
    create_manifest(args.training_manifest, positives, negatives)

    close_dbs()

def parse_args():
    parser = argparse.ArgumentParser(description="Creates a dataset")
    parser.add_argument("-r", "--root", default='data', 
                help="All filepaths are relative to the root")
    parser.add_argument("--sprite-in-file", default="data/original_sprites.txt", 
                help="File containing paths to each sprite")
    parser.add_argument("--sprite-out-dir", default="data/images/augmented_sprites",
                help="Output directory where the augmented sprites will be written")
    parser.add_argument("--sprites-no-cache", default=False, action="store_true",
                help="By default use any precomputed augmented sprites")
    parser.add_argument("-s", "--num-sprites", default=20, type=int, 
                help="Number of generated sprite images")

    parser.add_argument("-d", "--data-manifest", default='data/training_images.txt', 
                help="File containing paths to each training image")
    parser.add_argument("--crop-out-dir", default="data/images/random_crops",
                help="Output directory where the random crops are put")
    parser.add_argument("-n", "--num-crops", default=10000, type=int, 
                help="Number of total random crops")
    parser.add_argument("--crops-no-cache", default=False, action="store_true",
                help="By default use previously computed random crops")

    parser.add_argument("-p", "--percent-positive", default=0.5, type=float, 
                help="Percentage of random crops that will be positive examples")
    parser.add_argument("-v", "--percent-validation", default=0.1, type=float, 
                help="Percentage of random crops that will be used in validation set")
    parser.add_argument("-o", "--training-out-dir", default="data/images/training_data",
                help="Output directory where the random crops are put")
    parser.add_argument("-m", "--training-manifest", default="data/images/training_manifest.txt",
                help="Output Manifest File listing training images and labels")
    parser.add_argument("-l", "--lmdb-prefix", default="data/lmdb/waldo",
                help="Prefix for the lmdbs")

    parser.add_argument("-q", "--quiet", default=False, action="store_true",
                help="Supress Printing")

    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    QUIET = args.quiet
    main(args)

