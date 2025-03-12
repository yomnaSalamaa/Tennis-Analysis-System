from the_utils import read_video, save_video
from trackers import PlayerTracker 

def main():

    #Reading the video 
    input_video_path = 'input_videos/Tennis Video.mp4'
    video_frames, original_fps = read_video(input_video_path) 

    #Detecting players
    player_tracker =PlayerTracker(model_path = 'yolov8x') 
    player_detections = player_tracker.detect_frames(video_frames)

    #Draw output

    #Draw player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    save_video(output_video_frames, 'output_videos/output_video1.avi', original_fps)  # Pass FPS to save_video

if __name__ == '__main__':
    main()
