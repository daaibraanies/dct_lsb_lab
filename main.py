import sys

import numpy as np
from skimage.util import view_as_blocks
import skimage
from scipy.fftpack import idctn, dctn
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image


def read_img(path):
    img = Image.open(path).convert('L')
    img = np.array(img)
    return img


def eval_hologram(data):
    hologram = dctn(data, norm='ortho')
    min_h = data.min()
    max_h = data.max()
    
    min_condition = hologram < min_h
    max_condition = hologram > max_h
    other_condition = (min_h < hologram) & (hologram < max_h)
     
    hologram[min_condition] = 0
    hologram[max_condition] = 1 
    hologram[other_condition] = (hologram[other_condition] - min_h)/(max_h - min_h)

    return hologram


def restore_img(hologram):
    img = idctn(hologram, norm='ortho')
    img /= np.sum(img.shape)/2
    img = (img - img.min())/(img.max() - img.min())
    return img


def img_to_bits(img):
    img = np.uint8(img*255).flatten()
    # bits = np.unpackbits(img)
    return img


def bits_to_img(bits, w, h):
    values = np.packbits(bits)
    img = values.reshape((w, h))/255.
    return img 


u1, v1 = 4, 5
u2, v2 = 5, 4
n = 8
P = 25


def double_to_byte(arr):
    return np.uint8(np.round(np.clip(arr, 0, 255), 0))


def increment_abs(x):
    return x + 1 if x >= 0 else x - 1


def decrement_abs(x):
    if np.abs(x) <= 1:
        return 0
    else:
        return x - 1 if x >= 0 else x + 1
    

def abs_diff_coefs(transform):
    return abs(transform[u1, v1]) - abs(transform[u2, v2])



def embed_bit(block, bit):
    patch = block.copy()
    coefs = dctn(patch) 
    while not valid_coefficients(coefs, bit, P) or (bit != retrieve_bit(patch)):
        coefs = change_coefficients(coefs, bit)
        patch = double_to_byte(idctn(coefs)/(2*n)**2)
    return patch


def embed_message(orig, msg):
    changed = orig.copy()
    blocks = view_as_blocks(changed, block_shape=(n, n))
    h = blocks.shape[1]        
    for index, bit in enumerate(msg):
        i = index // h
        j = index % h
        block = blocks[i, j]
        changed[i*n: (i+1)*n, j*n: (j+1)*n] = embed_bit(block, bit)
    return changed


def retrieve_bit(block):
    transform = dctn(block)
    return 0 if abs_diff_coefs(transform) > 0 else 1


def retrieve_message(img, length):
    blocks = view_as_blocks(img, block_shape=(n, n))
    h = blocks.shape[1]
    return [retrieve_bit(blocks[index//h, index%h]) for index in range(length)]


def encode_lsb_image(image, depth, message):

    max_length = np.prod(image.shape + (depth,))
    message_length = message.shape[0]
    #assert message_length <= max_length, "Сообщение слишком длинное!"

    # binary_message = np.array(binary_message, dtype=np.uint8)
    binary_message = message
    binary_message = np.unpackbits(binary_message.reshape(binary_message.shape + (1,)), axis=binary_message.ndim)
    binary_message = np.append(binary_message.flatten(), [0]*(depth - message_length*8 % depth))
    binary_message = binary_message.reshape(-1, depth)

    binary_message = np.hstack((np.full(binary_message.shape[0]*(8-depth), 0).reshape(binary_message.shape[0],-1), binary_message[:]))
    # print(binary_message)

    # print(np.unpackbits(np.arange(0, 10, dtype=np.uint8).reshape(10,1), axis=1))
    binary_image = np.unpackbits(image.reshape(image.shape + (1,)), axis=image.ndim)

    image_shape = binary_image.shape
    binary_image = binary_image.reshape(-1, 8)
    bytewise_mask = np.tile(np.hstack((np.array([1]*(8-depth)), np.array([0]*(depth)))), (binary_message.shape[0], 1))
    # bytewise_mask = bytewise_mask.reshape(binary_image.shape[0])
    binary_image[:binary_message.shape[0]] = np.bitwise_or(np.bitwise_and(binary_image[:binary_message.shape[0]], bytewise_mask), binary_message)
    binary_image = binary_image.reshape(image_shape)
    # print(binary_image)

    new_image = np.packbits(binary_image, axis=binary_image.ndim -1).reshape(image.shape)

    return new_image

def decode_lsb_image(image, depth:int, message_length:int):
    binary_image = np.unpackbits(image.reshape(image.shape + (1,)), axis=image.ndim)

    message_bytes_length = 8*message_length // depth
    message_bytes_length += 1 if message_length*8 % depth else 0

    image_shape = binary_image.shape
    binary_image = binary_image.reshape(-1, 8)
    binary_message = binary_image[:message_bytes_length, -depth:].flatten()[:8*message_length].reshape(-1, 8)
    # binary_message = np.packbits(binary_message, axis=binary_image.ndim-1).reshape(-1)

    return np.packbits(binary_message, axis=binary_image.ndim-1).reshape(-1)    


if __name__ == '__main__':
    hologram_path = sys.argv[1]
    container_path = sys.argv[2]
    img = read_img(hologram_path)
    container = skimage.io.imread(container_path)

    hologram = eval_hologram(img)
    w, h = hologram.shape
    bits = img_to_bits(hologram)
    img_with_stego = encode_lsb_image(container, 1, bits)
    img = decode_lsb_image(img_with_stego, 1, w*h)
    restored_img = restore_img(img.reshape(w, h) /255.)
    scipy.misc.imsave('restored.bmp', restored_img) 
    scipy.misc.imsave('stego.bmp', img_with_stego) 
    scipy.misc.imsave('holo.bmp', hologram) 



