import cv2
import os

def select_point(event, x, y, flags, param):
    global point_selected, point, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Draw the point
        cv2.imshow("Frame", frame)

def process_video(input_video_path, output_text_path, screen_width=1366, screen_height=768):
    global point_selected, point, frame

    point_selected = False
    point = (0, 0)

    # Read the video
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video '{input_video_path}'")
        return

    # Get the frame dimensions
    frame_width = 1366
    frame_height = 768

    # Create a text file to store the points
    with open(output_text_path, "a") as file:
        cv2.namedWindow("Frame")
        ret, frame = cap.read()  # Read the first frame
        if not ret:
            print(f"Error: Unable to read first frame of video '{input_video_path}'")
            return

        # Resize the frame
        resized_frame = cv2.resize(frame, (screen_width, screen_height))
        cv2.setMouseCallback("Frame", select_point)

        print("Click on the frame to select a point and press ENTER.")
        while True:
            # Update the frame with the marked point
            if point_selected:
                cv2.circle(resized_frame, point, 5, (0, 255, 0), -1)  # Draw the point
            cv2.imshow("Frame", resized_frame)

            # Wait for Enter key (13) or Escape key (27) to break the loop
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                if point_selected:
                    filename = os.path.splitext(os.path.basename(input_video_path))[0]
                    file.write(f"{filename}: {point[0]}, {point[1]}\n")  # Write filename and point coordinates to file
                    print(f"Point saved for {filename}: {point}")
                else:
                    print("No point selected.")
                break
            elif key == 27:  # ESC key
                print("Selection cancelled.")
                break

    cap.release()
    cv2.destroyAllWindows()

def process_videos(input_dir, output_dir):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each video file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):  # Check if the file is a video file
            input_video_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, "PS2_points.txt")  # Combined output file for all videos
            process_video(input_video_path, output_text_path)

# Example usage:
input_path = r"C:\Users\ybr5070\Documents\PortionSize2"
output_path = r"C:\Users\ybr5070\Desktop\PS2_ROI_texts"

process_videos(input_path, output_path)
