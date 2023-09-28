# Instructions for preparing the datasets

## Download the datasets
1. Benchmark 1: [MSU-MFSD](https://sites.google.com/site/huhanhomepage/download/), [CASIA-MFSD](https://ieeexplore.ieee.org/document/6199754), [Replay-attack](https://www.idiap.ch/en/dataset/replayattack), [CelebA-Spoof](https://drive.google.com/corp/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z), [OULU-NPU](https://sites.google.com/site/oulunpudatabase/)
2. Benchmark 2: [WMCA](https://www.idiap.ch/en/dataset/wmca), [CASIA-SURF CeFA](https://sites.google.com/corp/qq.com/face-anti-spoofing/dataset-download/casia-surf-cefacvpr2020), [CASIA-SURF](https://sites.google.com/corp/qq.com/face-anti-spoofing/dataset-download/casia-surfcvpr2019)


## Data preprocessing - Benchmark 1
- For benchmark 1, follow the preprocessing step of [SSDG](https://github.com/taylover-pei/SSDG-CVPR2020) to detect and align the faces using [MTCNN](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection). 
1. For each video, only sample two frames: frame[6] and frame[6+math.floor(total_frames/2)] and save the frame as videoname_frame0.png/videoname_frame1.png, except for the CelebA-Spoof dataset.
2. Input the sample frames into MTCNN to detect, align and crop the images. The image are resized to (224,224,3), and only the RGB channels are used.
3. Save the frames into data/MCIO/frame/ following the file names listed in data/MCIO/txt/, with the following directory structure:

   ```
   data/MCIO/frame/
   |-- casia
       |-- train
       |   |--real
       |   |  |--1_1_frame0.png, 1_1_frame1.png 
       |   |--fake
       |      |--1_3_frame0.png, 1_3_frame1.png 
       |-- test
           |--real
           |  |--1_1_frame0.png, 1_1_frame1.png 
           |--fake
              |--1_3_frame0.png, 1_3_frame1.png 
   |-- msu
       |-- train
       |   |--real
       |   |  |--real_client002_android_SD_scene01_frame0.png, real_client002_android_SD_scene01_frame1.png
       |   |--fake
       |      |--attack_client002_android_SD_ipad_video_scene01_frame0.png, attack_client002_android_SD_ipad_video_scene01_frame1.png
       |-- test
           |--real
           |  |--real_client001_android_SD_scene01_frame0.png, real_client001_android_SD_scene01_frame1.png
           |--fake
              |--attack_client001_android_SD_ipad_video_scene01_frame0.png, attack_client001_android_SD_ipad_video_scene01_frame1.png
   |-- replay
       |-- train
       |   |--real
       |   |  |--real_client001_session01_webcam_authenticate_adverse_1_frame0.png, real_client001_session01_webcam_authenticate_adverse_1_frame1.png
       |   |--fake
       |      |--fixed_attack_highdef_client001_session01_highdef_photo_adverse_frame0.png, fixed_attack_highdef_client001_session01_highdef_photo_adverse_frame1.png
       |-- test
           |--real
           |  |--real_client009_session01_webcam_authenticate_adverse_1_frame0.png, real_client009_session01_webcam_authenticate_adverse_1_frame1.png
           |--fake
              |--fixed_attack_highdef_client009_session01_highdef_photo_adverse_frame0.png, fixed_attack_highdef_client009_session01_highdef_photo_adverse_frame1.png
   |-- oulu
       |-- train
       |   |--real
       |   |  |--1_1_01_1_frame0.png, 1_1_01_1_frame1.png
       |   |--fake
       |      |--1_1_01_2_frame0.png, 1_1_01_2_frame1.png
       |-- test
           |--real
           |  |--1_1_36_1_frame0.png, 1_1_36_1_frame1.png
           |--fake
              |--1_1_36_2_frame0.png, 1_1_36_2_frame1.png
   |-- celeb
       |-- real
       |   |--167_live_096546.jpg
       |-- fake
           |--197_spoof_420156.jpg       
   ```


## Data preprocessing - Benchmark 2
- For benchmark 2, use the original frames and cut the black borders.

1. Use all the frames in surf dataset with their original file names. Sample 10 frames in each video equidistantly for cefa and wmca datasets. Save the sampled frame as videoname_XX.jpg (where XX denotes the index of sampled frame). Detailed file names can be found in data/WCS/txt/.
2. [Cut the black borders](https://github.com/AlexanderParkin/CASIA-SURF_CeFA/blob/205d3d976523ed0c15d1e709ed7f21d50d7cf19b/at_learner_core/at_learner_core/utils/transforms.py#L456) of the images, which will be the input images. The images are then resized to (224,224,3) and only the RGB channels are used.
3. Save the frames into data/WCS/frame/ following the file names listed in data/WCS/txt/, with the following directory structure:
   
   ```
   data/WCS/frame/
   |-- wmca
       |-- train
       |   |--real
       |   |  |--31.01.18_035_01_000_0_01_00.jpg, 31.01.18_035_01_000_0_01_05.jpg
       |   |--fake
       |      |--31.01.18_514_01_035_1_05_00.jpg, 31.01.18_514_01_035_1_05_05.jpg
       |-- test
           |--real
           |  |--31.01.18_036_01_000_0_00_00.jpg, 31.01.18_036_01_000_0_00_01.jpg
           |--fake
              |--31.01.18_098_01_035_3_13_00.jpg, 31.01.18_098_01_035_3_13_01.jpg
   |-- cefa
       |-- train
       |   |--real
       |   |  |--3_499_1_1_1_00.jpg, 3_499_1_1_1_01.jpg
       |   |--fake
       |      |--3_499_3_2_2_00.jpg, 3_499_3_2_2_01.jpg
       |-- test
           |--real
           |  |--3_299_1_1_1_00.jpg, 3_299_1_1_1_01.jpg
           |--fake
              |--3_299_3_2_2_00.jpg, 3_299_3_2_2_01.jpg
   |-- surf
       |-- train
       |   |--real
       |   |  |--Training_real_part_CLKJ_CS0110_real.rssdk_color_91.jpg
       |   |--fake
       |      |--Training_fake_part_CLKJ_CS0110_06_enm_b.rssdk_color_91.jpg
       |-- test
           |--real
           |  |--Val_0007_007243-color.jpg
           |--fake
              |--Val_0007_007193-color.jpg
   ```

# Acknowledgement
The data-preprocessing method mentioned above is followed directly from [few-shot-fas](https://github.com/hhsinping/few_shot_fas) repository. We thank the authors for their great work and for making the code public.