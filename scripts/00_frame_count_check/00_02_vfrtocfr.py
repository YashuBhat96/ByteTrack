import os
import subprocess
import logging
from tqdm import tqdm
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_directory_exists(directory):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logging.info(f"Directory created: {directory}")
        except OSError as e:
            logging.error(f"Failed to create directory {directory}: {str(e)}")
            return False
    return True

def convert_video_to_cfr(video_path, ffmpeg_path, output_directory, output_video_fps, threads):
    """Convert a video to a constant frame rate (CFR)."""
    output_video_path = os.path.join(output_directory, os.path.basename(video_path).replace('.avi', '_fixed.mp4').replace('.mp4', '_fixed.mp4'))

    command = [
        ffmpeg_path, '-loglevel', 'verbose', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda', '-i', video_path,
        '-r', str(output_video_fps), '-c:v', 'h264_nvenc', '-an', output_video_path
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logging.error(f"FFmpeg failed for {video_path} with the following output:")
            logging.error(result.stdout)
            logging.error(result.stderr)
        else:
            logging.info(f"Successfully converted: {video_path} to {output_video_path}")
    except Exception as e:
        logging.error(f"An error occurred while processing {video_path}: {str(e)}")

def process_all_videos(input_directory, ffmpeg_path, output_directory, output_fps, max_threads, max_workers):
    """Process all videos in a directory."""
    if not ensure_directory_exists(output_directory):
        return

    video_files = [f for f in os.listdir(input_directory) if f.endswith(('.avi', '.mp4', '.mov'))]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_video_to_cfr, os.path.join(input_directory, video_file), ffmpeg_path, output_directory, output_fps, max_threads): video_file for video_file in video_files}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing videos", unit="video"):
            try:
                future.result()
            except Exception as e:
                video_file = futures[future]
                logging.error(f"Error processing {video_file}: {str(e)}")

# Configuration for converting video files
ffmpeg_path = "/swst/apps/ffmpeg/4.3.2_gcc-8.5.0/bin/ffmpeg"
input_directory = "/storage/group/klk37/default/homebytes/video/fbs/PS_vids_original/PortionSize2/"
output_directory = "/storage/group/klk37/default/homebytes/video/fbs/PS2_vids_fixed/"
output_fps = 30
max_threads = 16
max_workers = 4  # Number of parallel workers

process_all_videos(input_directory, ffmpeg_path, output_directory, output_fps, max_threads, max_workers)


ffmpeg_path, '-loglevel', 'verbose', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda', '-i', video_path,
        '-r', str(output_video_fps), '-c:v', 'h264_nvenc', '-an', output_video_path