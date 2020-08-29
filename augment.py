from PIL import Image
import random
import math
import numbers

def group_multi_scale_crop(img_group, target_size, scales=None, \
                           max_distort=1, fix_crop=True, more_fix_crop=True):
    scales = scales if scales is not None else [1, .875, .75, .66]
    input_size = [target_size, target_size]

    im_size = img_group[0].size

    # get random crop offset
    def _sample_crop_size(im_size):
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in scales]
        crop_h = [
            input_size[1] if abs(x - input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            input_size[0] if abs(x - input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_step = (image_w - crop_pair[0]) / 4
            h_step = (image_h - crop_pair[1]) / 4

            ret = list()
            ret.append((0, 0))  # upper left
            if w_step != 0:
                ret.append((4 * w_step, 0))  # upper right
            if h_step != 0:
                ret.append((0, 4 * h_step))  # lower left
            if h_step != 0 and w_step != 0:
                ret.append((4 * w_step, 4 * h_step))  # lower right
            if h_step != 0 or w_step != 0:
                ret.append((2 * w_step, 2 * h_step))  # center

            if more_fix_crop:
                ret.append((0, 2 * h_step))  # center left
                ret.append((4 * w_step, 2 * h_step))  # center right
                ret.append((2 * w_step, 4 * h_step))  # lower center
                ret.append((2 * w_step, 0 * h_step))  # upper center

                ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

            w_offset, h_offset = random.choice(ret)

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
    crop_img_group = [
        img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        for img in img_group
    ]
    ret_img_group = [
        img.resize((input_size[0], input_size[1]), Image.BILINEAR)
        for img in crop_img_group
    ]

    return ret_img_group


def get_params(img, scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.)): # scale 0.25 1.0
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        img (PIL Image): Image to be cropped.
        scale (tuple): range of size of the origin size cropped
        ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    area = img.size[0] * img.size[1]

    for attempt in range(10):
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5 and min(ratio) <= (h / w) <= max(ratio):
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            i = random.randint(0, img.size[1] - h)
            j = random.randint(0, img.size[0] - w)
            return i, j, h, w

    # Fallback
    w = min(img.size[0], img.size[1])
    i = (img.size[1] - w) // 2
    j = (img.size[0] - w) // 2
    return i, j, w, w
def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    # if not _is_pil_image(img):
    #     raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    # if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
    #     raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def group_random_crop1(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
        "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images

def group_random_crop(img_group, target_size):

    out_images = []
    i, j, h, w = get_params(img_group[0])

    for img in img_group:
        img = img.crop((j, i, j + w, i + h))
        img = resize(img, (target_size, target_size))
        out_images.append(img)

    return out_images


def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group



def crop(img, i, j, h, w):
    """Crop the given PIL Image.

    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))

def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        img_crop.append(center_crop(img, target_size))
    return img_crop


def group_scale(imgs, target_size):
    resized_imgs = []
    for img in imgs:
        resized_imgs.append(resize(img, target_size))
    return resized_imgs

class CornerCrop(object):

    def __init__(self,
                 size,
                 crop_position=None,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.size = size
        self.crop_position = crop_position
        self.crop_positions = crop_positions

        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.randomize_parameters()

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        h, w = (self.size, self.size / image_height * image_width)
        if self.crop_position == 'c':
            i = int(round((image_height - h) / 2.))
            j = int(round((image_width - w) / 2.))
        elif self.crop_position == 'tl':
            i = 0
            j = 0
        elif self.crop_position == 'tr':
            i = 0
            j = image_width - w
        elif self.crop_position == 'bl':
            i = image_height - h
            j = 0
        elif self.crop_position == 'br':
            i = image_height - h
            j = image_width - w

        img = crop(img, i, j, h, w)

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, crop_position={1}, randomize={2})'.format(
            self.size, self.crop_position, self.randomize)


def group_random_corner_crop(imgs, size, crop_positions=('c', 'tl', 'tr', 'bl', 'br')):
    scales = [1.0]
    scale_step = 1 / (2 ** (1 / 4))
    for _ in range(1, 5):
        scales.append(scales[-1] * scale_step)
    scale = scales[random.randint(0, len(scales) - 1)]
    crop_position = crop_positions[random.randint(
        0,
        len(crop_positions) - 1)]

    corner_crop = CornerCrop(None, crop_position)
    resized_images = []
    for img in imgs:
        short_side = min(img.size[0], img.size[1])
        crop_size = int(short_side * scale)
        corner_crop.size = crop_size

        img = corner_crop(img)
        # resized_images.append(img.resize((size, size), Image.ANTIALIAS))
        resized_images.append(resize(img, size))
    return resized_images

def MultiScaleRandomCrop(imgs, scales, size, interpolation=Image.BILINEAR):
    scale = scales[random.randint(0, len(scales) - 1)]
    tl_x = random.random()
    tl_y = random.random()
    img = imgs[0]
    min_length = min(img.size[0], img.size[1])
    crop_size = int(min_length * scale)

    image_width = img.size[0]
    image_height = img.size[1]

    x1 = tl_x * (image_width - crop_size)
    y1 = tl_y * (image_height - crop_size)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    resized_images = []
    for img in imgs:
        img = img.crop((x1, y1, x2, y2))
        resized_images.append(img.resize((size, size), interpolation))
    return resized_images
