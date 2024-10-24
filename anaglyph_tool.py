from tqdm import tqdm
import argparse

import cv2
import numpy as np
import sys
import os
import torch
from PIL import Image

class DepthModel:
    def __init__(self, encoder):
        try:
            print("Loading depth model...")
            depth_estimation_path = os.path.join(os.getcwd(), 'Depth-Anything-V2')
            sys.path.append(depth_estimation_path)
            from depth_anything_v2.dpt import DepthAnythingV2
            
            DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            
            
            model = DepthAnythingV2(**model_configs[encoder])
            model.load_state_dict(torch.load(f'./Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
            self.model = model.to(DEVICE).eval()
            print("Depth model loaded.")
        except Exception as e:
            print("Error while loading model: ", e)
            quit()
        
    def estimate_depth(self, image):
        return self.model.infer_image(image)

def process_frame(model, frame, displacement, filter_kernel):
    """
    Makes image stereoscopic
    """
    blue_channel, green_channel, red_channel = cv2.split(frame)
    height, width = red_channel.shape

    depth_map = model.estimate_depth(frame) 
    depth_map/=depth_map.max()
    depth_map = cv2.filter2D(depth_map,-1,filter_kernel)

        
    displacement_map_red = -displacement * depth_map 
    displacement_map_cyan = displacement * depth_map 

    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    shifted_x_coords_red = (x_coords + displacement_map_red).astype(np.float32)
    shifted_red_channel = cv2.remap(red_channel, shifted_x_coords_red, y_coords.astype(np.float32), cv2.INTER_LINEAR)

    shifted_x_coords_cyan = (x_coords + displacement_map_cyan).astype(np.float32)
    shifted_blue_channel = cv2.remap(blue_channel, shifted_x_coords_cyan, y_coords.astype(np.float32), cv2.INTER_LINEAR)
    shifted_green_channel = cv2.remap(green_channel, shifted_x_coords_cyan, y_coords.astype(np.float32), cv2.INTER_LINEAR)

    return cv2.merge([shifted_blue_channel, shifted_green_channel, shifted_red_channel]), depth_map



def main():
    parser = argparse.ArgumentParser(
        description="Converts images and videos to anaglyph."
    )

    parser.add_argument(
        "-i", "--input",
        type=str, 
        help="Input file path (image or video)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str, 
        default='vitb',
        help="DepthAnythingV2 model to use"
    )
    parser.add_argument(
        "-s", "--shift", 
        type=int, 
        default=15,
        help="Scaling of red and cyan channel displacement. Higher values make scene seem smaller."
    )
    parser.add_argument(
        "-f", "--frames",
        type=int, 
        default=-1,
        help="Maximum number of frames to process."
    )
    parser.add_argument(
        "-o", "--output",
        type=str, 
        default='output',
        help="Output file name"
    )
    
    args = parser.parse_args()


    

    input_path = args.input
    extension = input_path[input_path.rfind('.')+1:].lower()
    displacement = args.shift 

    output_path = args.output

    ds= 5
    filter_kernel = np.ones((ds,ds),np.float32)/(ds**2)

    if extension in ['jpg', 'png', 'jpeg']:  #process image
       
        frame = np.array(Image.open(input_path))
        try:
            if frame.shape[2] == 4:
                frame = frame[:,:,:3]
            if frame.shape[2] == 1:
                raise Exception("Error: Image is not RGB.")
            frame = np.flip(frame,axis=2)
        except Exception as e:
            print(e)
            quit()

        dm = DepthModel(args.model)
        frame, depth_map = process_frame(dm, frame, displacement, filter_kernel)
        frame = np.flip(frame,axis=2)
        image_pil = Image.fromarray(frame)
        final_output = output_path+ '.' + extension
        image_pil.save(final_output)
        print("Success. Final image saved as: ", final_output)

    else: #process video
        
        dm = DepthModel(args.model)

        cap = cv2.VideoCapture(input_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        temp_video_output = output_path + '_no_audio.avi' 
        out = cv2.VideoWriter(temp_video_output, fourcc, fps, (frame_width, frame_height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.frames == -1 else args.frames
        frame_index = 0

        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame, depth_map = process_frame(dm, frame, displacement, filter_kernel)
                out.write(frame)

                pbar.update(1)
                frame_index += 1

                if frame_index == total_frames:
                    break

        cap.release()
        out.release()

        processed_video_duration = args.frames / fps 
        audio_output = output_path + '_audio.aac'
        os.system(f'ffmpeg -i "{input_path}" -vn -acodec copy -t {processed_video_duration} "{audio_output}"')
        final_output = output_path + '.mp4'
        os.system(f'ffmpeg -i "{temp_video_output}" -i "{audio_output}" -c:v copy -c:a aac -strict experimental "{final_output}"')
        os.remove(temp_video_output)
        os.remove(audio_output)

        print("Success. Final video saved as: ", final_output)

if __name__ == '__main__':
    main()
