import cv2
import os
import logging
import concurrent.futures

# Setup logging
logging.basicConfig(filename='/storage/home/ybr5070/group/homebytes/code/scripts/LSTM_trial3/logs/frame_extraction.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def extract_frames_from_video(video_path, output_dir):
    vidcap = cv2.VideoCapture(video_path)
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.warning(f"Skipping video {video_path} due to invalid FPS")
        return
    
    frame_interval = int(fps / 6)
    if frame_interval == 0:
        logging.warning(f"Skipping video {video_path} due to low FPS")
        return

    video_name = os.path.basename(video_path).split('.')[0]
    frame_save_path = os.path.join(output_dir, video_name)
    os.makedirs(frame_save_path, exist_ok=True)
    
    success, image = vidcap.read()
    count = 0
    frame_count = 0
    while success:
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(frame_save_path, f"frame{count}.jpg")
            cv2.imwrite(frame_filename, image)
            count += 1

        success, image = vidcap.read()
        frame_count += 1

    logging.info(f"Finished extracting frames from {video_path}")

def main():
    # List of input directories for different portions
    portion_dirs = {
        "PS1": "/storage/group/klk37/default/homebytes/video/fbs/PS_fixed/PS1_fixed",
        "PS2": "/storage/group/klk37/default/homebytes/video/fbs/PS_fixed/PS2_fixed",
        "PS3": "/storage/group/klk37/default/homebytes/video/fbs/PS_fixed/PS3_fixed",
        "PS4": "/storage/group/klk37/default/homebytes/video/fbs/PS_fixed/PS4_fixed"
    }
    
    # Output directory base
    output_base_dir = "/storage/group/klk37/default/homebytes/video/fbs/all_frames"

    # Create output directories for each portion size
    for ps_label in portion_dirs.keys():
        os.makedirs(os.path.join(output_base_dir, ps_label), exist_ok=True)

    # Using ThreadPoolExecutor to process video files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for ps_label, input_dir in portion_dirs.items():
            # Fetching all video files in the current input directory
            video_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) for f in filenames if f.endswith(('.mp4', '.avi'))]
            
            # Defining the output directory for the current portion size
            output_dir = os.path.join(output_base_dir, ps_label)
            
            # Submitting tasks to the executor
            for video_file in video_files:
                futures.append(executor.submit(extract_frames_from_video, video_file, output_dir))
        
        # Ensuring all futures are completed
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing video: {e}")

    logging.info("All videos processed!")

if __name__ == "__main__":
    main()
