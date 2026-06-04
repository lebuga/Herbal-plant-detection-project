import cv2
import os

video_folder = r"D:\Herbal_Plant_Project\videos"
output_base = r"D:\Herbal_Plant_Project\dataset"

FRAME_SKIP = 10  # adjust if needed

# Loop through all videos
for video_file in os.listdir(video_folder):

    if not video_file.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(video_folder, video_file)

    # Use video name (without extension) as class label
    class_name = os.path.splitext(video_file)[0]

    output_dir = os.path.join(output_base, class_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open: {video_file}")
        continue

    frame_count = 0
    saved_count = 0

    print(f"\n🎥 Processing: {video_file}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            save_path = os.path.join(
                output_dir,
                f"{class_name}_{saved_count:04d}.jpg"
            )
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"✅ Saved {saved_count} frames for {class_name}")

print("\n🎉 ALL VIDEOS PROCESSED SUCCESSFULLY")