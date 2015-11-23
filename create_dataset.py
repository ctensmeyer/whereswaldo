
import os
import sys
import glob
import random
import shutil
import argparse
import numpy as np
import warnings

from skimage import io, transform, filters

SPRITE_OCCLUSION_HEIGHT = 0.75
MIN_OCCLUSION_BB_AREA = 0.1
MAX_OCCLUSION_BB_AREA = 1

IMAGE_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 2]
CROP_SIZE = (170, 85)

QUIET = False

#Read in file of filenames
def loadFileNames(txt_file, root="."):
    names = []
    with open(txt_file, 'r') as f:
        for line in f:
            names.append(os.path.join(root, line.strip()))

    return names

def randomBeta(low, high, beta,alpha):
    return (random.betavariate(alpha,beta)* (high-low)) + low


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
        threshold = filters.threshold_otsu(sprite[:,:,3])
        sprite[sprite[:,:,3] < threshold] = 0
        sprite[sprite[:,:,3] >= threshold, 3] = 255

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
            sprite_sh = sprite_im.shape

            x = random.randrange(0, CROP_SIZE[1] - sprite_sh[1])
            y = random.randrange(0, CROP_SIZE[0] - sprite_sh[0])

            xEnd = x + sprite_sh[1]
            yEnd = y + sprite_sh[0]

            # insert sprite into crop
            inserted = insert(sprite_im, crop_im, x, y, xEnd, yEnd)

            imgLoc = os.path.join(positive_dir, "waldo_%d.png" % len(positive_files))
            io.imsave(imgLoc, inserted)
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
    sprites = getImages(args.num_sprites, args.root, args.sprite_in_file, args.sprite_out_dir, args.sprite_no_cache, createSprites, "sprites")
    crops = getImages(args.num_crops, args.root, args.data_manifest, args.crop_out_dir, args.crops_no_cache, createCrops, "crops")
    positives, negatives = make_instances(sprites, crops, args.root, args.training_out_dir, args.percent_positive)
    create_manifest(args.training_manifest, positives, negatives)

def parse_args():
    parser = argparse.ArgumentParser(description="Creates a dataset")
    parser.add_argument("-r", "--root", default='data', 
                help="All filepaths are relative to the root")
    parser.add_argument("--sprite-in-file", default="data/original_sprites.txt", 
                help="File containing paths to each sprite")
    parser.add_argument("--sprite-out-dir", default="data/images/augmented_sprites",
                help="Output directory where the augmented sprites will be written")
    parser.add_argument("--sprite-no-cache", default=False, action="store_true",
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
    parser.add_argument("-o", "--training-out-dir", default="data/images/training_data",
                help="Output directory where the random crops are put")
    parser.add_argument("-m", "--training-manifest", default="data/images/training_manifest.txt",
                help="Output Manifest File listing training images and labels")

    parser.add_argument("-q", "--quiet", default=False, action="store_true",
                help="Supress Printing")

    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    QUIET = args.quiet
    main(args)

