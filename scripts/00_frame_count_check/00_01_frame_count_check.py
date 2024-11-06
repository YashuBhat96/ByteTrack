import os
import subprocess
import logging
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_frame_count(video_path, ffmpeg_path):
    command = [
        ffmpeg_path, '-i', video_path, '-map', '0:v:0', '-c', 'copy', '-f', 'null', '-'
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    stderr_output = result.stderr

    for line in stderr_output.split('\n'):
        if 'frame=' in line:
            frame_info = line.split('frame=')[-1].strip()
            frame_count = int(frame_info.split()[0])
            return frame_count

    return None

def process_video_pair(video_path1, video_path2, ffmpeg_path):
    frame_count1 = get_frame_count(video_path1, ffmpeg_path)
    frame_count2 = get_frame_count(video_path2, ffmpeg_path)

    if frame_count1 != frame_count2:
        return {
            'video1': {'path': video_path1, 'frame_count': frame_count1},
            'video2': {'path': video_path2, 'frame_count': frame_count2}
        }
    return None

def process_and_compare(directory1, directory2, ffmpeg_path):
    video_files1 = {file for file in os.listdir(directory1) if file.endswith('.avi')}
    video_files2 = {file for file in os.listdir(directory2) if file.endswith('.mp4')}

    video_pairs = []
    mismatches = []

    for video_file in video_files1:
        base_filename = os.path.splitext(video_file)[0]
        matching_file = base_filename + '_fixed.mp4'

        if matching_file in video_files2:
            video_pairs.append((os.path.join(directory1, video_file), os.path.join(directory2, matching_file)))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_video_pair, pair[0], pair[1], ffmpeg_path): pair for pair in video_pairs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                mismatches.append(result)

            video_pair = futures[future]
            logging.info(f"Processed pair: {video_pair[0]}, {video_pair[1]}")

    if mismatches:
        for mismatch in mismatches:
            logging.info("Mismatch found:")
            logging.info(f"Video 1: {mismatch['video1']['path']}, Frame count: {mismatch['video1']['frame_count']}")
            logging.info(f"Video 2: {mismatch['video2']['path']}, Frame count: {mismatch['video2']['frame_count']}")
    else:
        logging.info("No mismatches found.")

if __name__ == "__main__":
    ffmpeg_path = "/swst/apps/ffmpeg/4.3.2_gcc-8.5.0/bin/ffmpeg"
    directory1_path = "/storage/group/klk37/default/homebytes/video/fbs/PS_vids_original/PortionSize1"
    directory2_path = "/storage/group/klk37/default/homebytes/video/fbs/PS_all_vids_30fps_fixed/PS1_vids_fixed"
    process_and_compare(directory1_path, directory2_path, ffmpeg_path)
