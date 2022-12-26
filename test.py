"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

def makeVideo2(result_frames, output_filename, framerate):
    # result_frames: list of numpy arrays (frame)
    print('Stack transferred frames back to video...')
    height,width,dim = result_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(os.path.join(output_filename,'transfer.mp4'),fourcc,framerate,(width,height))
    for j in range(len(result_frames)):
        frame = result_frames[j] * 255.0
        frame = frame.astype('uint8')
        video.write(frame)
    video.release()
    print('Transferred video saved at %s.'%output_filename)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    ######## use video transfer ########
    use_video = opt.use_video
    if use_video:
        #### read video, save each frame to ./temp_vid_frame/ ###
        temp_vid_frame_path = './temp_vid_frame/'
        if not os.path.exists(temp_vid_frame_path):
            os.makedirs(temp_vid_frame_path)
        temp_dir = glob.glob(temp_vid_frame_path+'*')
        # make sure to remove previous video's frame
        for frames in temp_dir: 
            os.remove(frames)

        caps = cv2.VideoCapture(opt.video_name)
        framerate = int(caps.get(5))

        success, frame = caps.read()
        count = 0
        while success:
            cv2.imwrite(f"{temp_vid_frame_path}{format(count,'05d')}.jpg", frame)     # save frame as JPEG file      
            success, frame = caps.read()
            count += 1
            print(f'total frame read is {count}', end='\r')
        
        opt.content_path = temp_vid_frame_path
        opt.num_test = count
    ########################################

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    if not use_video:
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    video_frame = []
    for i, data in enumerate(dataset):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
            # break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        if use_video:
            visual_np = visuals['cs'].cpu().detach().numpy()
            visual_np = np.transpose(np.squeeze(visual_np), (1,2,0))
            print(f'processing {i}-th frame', end='\r')
            video_frame.append(visual_np)

        img_path = model.get_image_paths()     # get image paths`
        if not use_video:
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, width=opt.display_winsize)
    if not use_video:
        webpage.save()  # save the HTML
    
    print(f'framerate is {framerate}')
    makeVideo2(video_frame, './output_transfer_video/' , framerate)