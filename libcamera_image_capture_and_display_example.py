# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2022, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>
# A simple libcamera capture example with image display

''' The code does not belong to me I just reused few components to achieve the output I needed.
The orignal code can be fond here https://github.com/kbingham/libcamera/tree/master/src/py 

Opencv is used to diplay the image but you may use any library you see fit as the image_array is numpy
which is compatible with a lot of libraries.
'''

import argparse
import libcamera as libcam
import selectors
import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Tuple
from cv2 import imshow, waitKey

TOTAL_FRAMES = 30

class MappedFrameBuffer:
    """
    Provides memoryviews for the FrameBuffer's planes
    """
    def __init__(self, fb: libcam.FrameBuffer):
        self.__fb = fb
        self.__planes = ()
        self.__maps = ()

    def __enter__(self):
        return self.mmap()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.munmap()

    def mmap(self):
        if self.__planes:
            raise RuntimeError('MappedFrameBuffer already mmapped')

        import os
        import mmap

        fb = self.__fb

        # Collect information about the buffers

        bufinfos = {}

        for plane in fb.planes:
            fd = plane.fd

            if fd not in bufinfos:
                buflen = os.lseek(fd, 0, os.SEEK_END)
                bufinfos[fd] = {'maplen': 0, 'buflen': buflen}
            else:
                buflen = bufinfos[fd]['buflen']

            if plane.offset > buflen or plane.offset + plane.length > buflen:
                raise RuntimeError(f'plane is out of buffer: buffer length={buflen}, ' +
                                   f'plane offset={plane.offset}, plane length={plane.length}')

            bufinfos[fd]['maplen'] = max(bufinfos[fd]['maplen'], plane.offset + plane.length)

        # mmap the buffers

        maps = []

        for fd, info in bufinfos.items():
            map = mmap.mmap(fd, info['maplen'], mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            info['map'] = map
            maps.append(map)

        self.__maps = tuple(maps)

        # Create memoryviews for the planes

        planes = []

        for plane in fb.planes:
            fd = plane.fd
            info = bufinfos[fd]

            mv = memoryview(info['map'])

            start = plane.offset
            end = plane.offset + plane.length

            mv = mv[start:end]

            planes.append(mv)

        self.__planes = tuple(planes)

        return self

    def munmap(self):
        if not self.__planes:
            raise RuntimeError('MappedFrameBuffer not mmapped')

        for p in self.__planes:
            p.release()

        for mm in self.__maps:
            mm.close()

        self.__planes = ()
        self.__maps = ()

    @property
    def planes(self) -> Tuple[memoryview, ...]:
        """memoryviews for the planes"""
        if not self.__planes:
            raise RuntimeError('MappedFrameBuffer not mmapped')

        return self.__planes

    @property
    def fb(self):
        return self.__fb



def demosaic(data, r0, g0, g1, b0):
    # Separate the components from the Bayer data to RGB planes

    rgb = np.zeros(data.shape + (3,), dtype=data.dtype)
    rgb[r0[1]::2, r0[0]::2, 0] = data[r0[1]::2, r0[0]::2]  # Red
    rgb[g0[1]::2, g0[0]::2, 1] = data[g0[1]::2, g0[0]::2]  # Green
    rgb[g1[1]::2, g1[0]::2, 1] = data[g1[1]::2, g1[0]::2]  # Green
    rgb[b0[1]::2, b0[0]::2, 2] = data[b0[1]::2, b0[0]::2]  # Blue

    # Below we present a fairly naive de-mosaic method that simply
    # calculates the weighted average of a pixel based on the pixels
    # surrounding it. The weighting is provided by a byte representation of
    # the Bayer filter which we construct first:

    bayer = np.zeros(rgb.shape, dtype=np.uint8)
    bayer[r0[1]::2, r0[0]::2, 0] = 1  # Red
    bayer[g0[1]::2, g0[0]::2, 1] = 1  # Green
    bayer[g1[1]::2, g1[0]::2, 1] = 1  # Green
    bayer[b0[1]::2, b0[0]::2, 2] = 1  # Blue

    # Allocate an array to hold our output with the same shape as the input
    # data. After this we define the size of window that will be used to
    # calculate each weighted average (3x3). Then we pad out the rgb and
    # bayer arrays, adding blank pixels at their edges to compensate for the
    # size of the window when calculating averages for edge pixels.

    output = np.empty(rgb.shape, dtype=rgb.dtype)
    window = (3, 3)
    borders = (window[0] - 1, window[1] - 1)
    border = (borders[0] // 2, borders[1] // 2)

    rgb = np.pad(rgb, [
        (border[0], border[0]),
        (border[1], border[1]),
        (0, 0),
    ], 'constant')
    bayer = np.pad(bayer, [
        (border[0], border[0]),
        (border[1], border[1]),
        (0, 0),
    ], 'constant')

    # For each plane in the RGB data, we use a nifty numpy trick
    # (as_strided) to construct a view over the plane of 3x3 matrices. We do
    # the same for the bayer array, then use Einstein summation on each
    # (np.sum is simpler, but copies the data so it's slower), and divide
    # the results to get our weighted average:

    for plane in range(3):
        p = rgb[..., plane]
        b = bayer[..., plane]
        pview = as_strided(p, shape=(
            p.shape[0] - borders[0],
            p.shape[1] - borders[1]) + window, strides=p.strides * 2)
        bview = as_strided(b, shape=(
            b.shape[0] - borders[0],
            b.shape[1] - borders[1]) + window, strides=b.strides * 2)
        psum = np.einsum('ijkl->ij', pview)
        bsum = np.einsum('ijkl->ij', bview)
        output[..., plane] = psum // bsum

    return output


def to_rgb(fmt, size, data):
    w = size.width
    h = size.height
    if fmt == libcam.formats.YUYV:
        # YUV422
        yuyv = data.reshape((h, w // 2 * 4))

        # YUV444
        yuv = np.empty((h, w, 3), dtype=np.uint8)
        yuv[:, :, 0] = yuyv[:, 0::2]                    # Y
        yuv[:, :, 1] = yuyv[:, 1::4].repeat(2, axis=1)  # U
        yuv[:, :, 2] = yuyv[:, 3::4].repeat(2, axis=1)  # V

        m = np.array([
            [1.0, 1.0, 1.0],
            [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
            [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]
        ])

        rgb = np.dot(yuv, m)
        rgb[:, :, 0] -= 179.45477266423404
        rgb[:, :, 1] += 135.45870971679688
        rgb[:, :, 2] -= 226.8183044444304
        rgb = rgb.astype(np.uint8)

    elif fmt == libcam.formats.RGB888:
        rgb = data.reshape((h, w, 3))
        rgb[:, :, [0, 1, 2]] = rgb[:, :, [2, 1, 0]]

    elif fmt == libcam.formats.BGR888:
        rgb = data.reshape((h, w, 3))

    elif fmt in [libcam.formats.ARGB8888, libcam.formats.XRGB8888]:
        rgb = data.reshape((h, w, 4))
        rgb = np.flip(rgb, axis=2)
        # drop alpha component
        rgb = np.delete(rgb, np.s_[0::4], axis=2)

    elif str(fmt).startswith('S'):
        fmt = str(fmt)
        bayer_pattern = fmt[1:5]
        bitspp = int(fmt[5:])

        # \todo shifting leaves the lowest bits 0
        if bitspp == 8:
            data = data.reshape((h, w))
            data = data.astype(np.uint16) << 8
        elif bitspp in [10, 12]:
            data = data.view(np.uint16)
            data = data.reshape((h, w))
            data = data << (16 - bitspp)
        else:
            raise Exception('Bad bitspp:' + str(bitspp))

        idx = bayer_pattern.find('R')
        assert(idx != -1)
        r0 = (idx % 2, idx // 2)

        idx = bayer_pattern.find('G')
        assert(idx != -1)
        g0 = (idx % 2, idx // 2)

        idx = bayer_pattern.find('G', idx + 1)
        assert(idx != -1)
        g1 = (idx % 2, idx // 2)

        idx = bayer_pattern.find('B')
        assert(idx != -1)
        b0 = (idx % 2, idx // 2)

        rgb = demosaic(data, r0, g0, g1, b0)
        rgb = (rgb >> 8).astype(np.uint8)

    else:
        rgb = None

    return rgb


# A naive format conversion to 24-bit RGB
def mfb_to_rgb(mfb: MappedFrameBuffer, cfg: libcam.StreamConfiguration):
    data = np.array(mfb.planes[0], dtype=np.uint8)
    rgb = to_rgb(cfg.pixel_format, cfg.size, data)
    return rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', type=str, default='1',
                        help='Camera index number (starting from 1) or part of the name')
    parser.add_argument('-f', '--format', type=str, help='Pixel format')
    parser.add_argument('-s', '--size', type=str, help='Size ("WxH")')
    args = parser.parse_args()

    cm = libcam.CameraManager.singleton()

    try:
        if args.camera.isnumeric():
            cam_idx = int(args.camera)
            cam = next((cam for i, cam in enumerate(cm.cameras) if i + 1 == cam_idx))
        else:
            cam = next((cam for cam in cm.cameras if args.camera in cam.id))
    except Exception:
        print(f'Failed to find camera "{args.camera}"')
        return -1
    ret = cam.acquire()
    assert ret == 0
    cam_config = cam.generate_configuration([libcam.StreamRole.Viewfinder])

    stream_config = cam_config.at(0)
    
    #Set the foramt manually in case the code isn't picking it up, I had to set it manually as I was having some issues
    #stream_config.pixel_format = libcam.formats.YUYV

    if args.format:
        fmt = libcam.PixelFormat(args.format)
        stream_config.pixel_format = fmt

    if args.size:
        w, h = [int(v) for v in args.size.split('x')]
        stream_config.size = libcam.Size(w, h)

    ret = cam.configure(cam_config)
    assert ret == 0

    print(f'Capturing {TOTAL_FRAMES} frames with {stream_config}')

    stream = stream_config.stream
    allocator = libcam.FrameBufferAllocator(cam)
    ret = allocator.allocate(stream)
    assert ret > 0

    num_bufs = len(allocator.buffers(stream))
    reqs = []
    for i in range(num_bufs):
        req = cam.create_request(i)
        buffer = allocator.buffers(stream)[i]
        ret = req.add_buffer(stream, buffer)
        assert ret == 0

        reqs.append(req)
    ret = cam.start()
    assert ret == 0
    frames_queued = 0
    frames_done = 0
    for req in reqs:
        ret = cam.queue_request(req)
        assert ret == 0
        frames_queued += 1
    sel = selectors.DefaultSelector()
    sel.register(cm.event_fd, selectors.EVENT_READ)

    while frames_done < TOTAL_FRAMES:
        events = sel.select()
        if not events:
            continue

        reqs = cm.get_ready_requests()

        for req in reqs:
            frames_done += 1

            buffers = req.buffers
            assert len(buffers) == 1

            stream, fb = next(iter(buffers.items()))
            meta = fb.metadata
            mfb = MappedFrameBuffer(fb).mmap()
            image_array = mfb_to_rgb(mfb, stream.configuration)
            imshow("image",image_array)
            if waitKey(25) & 0xFF == ord('q'):
                break
            print("seq {:3}, bytes {}, frames queued/done {:3}/{:<3}"
                  .format(meta.sequence,
                          '/'.join([str(p.bytes_used) for p in meta.planes]),
                          frames_queued, frames_done))
            if frames_queued < TOTAL_FRAMES:
                req.reuse()
                cam.queue_request(req)
                frames_queued += 1
    ret = cam.stop()
    assert ret == 0
    ret = cam.release()
    assert ret == 0
    return 0

if __name__ == '__main__':
    sys.exit(main())


