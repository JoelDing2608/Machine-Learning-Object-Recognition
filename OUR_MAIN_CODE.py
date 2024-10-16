"""
Deep Learning Project
Object Detection on Live Video, MP4 Video and Online Video

"""
import time
from pathlib import Path
import cv2
import torch
from numpy import random
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from pytube import YouTube

################################################################

def get_youtube_stream_url(youtube_url):
    # Fetch the first available progressive stream URL from YouTube video
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    return stream.url

def detect_real_time(weights='yolov7.pt', source='0', conf_thres=0.25, iou_thres=0.45, imgsz=448, min_confidence=0.5):
    view_img = True  # Display results
    webcam = source.isnumeric() or source.lower().startswith(('http://', 'https://'))

    # Initialize device to use (GPU if available, otherwise CPU)
    device = select_device('')
    half = device.type != 'cpu'  # Half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # Set DataLoader
    cudnn.benchmark = True  # Set True to speed up constant image size inference

    if webcam or source.lower().startswith(('http://', 'https://')):
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get class names and random colors for bounding boxes
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names] # random colors for each object

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0  # Convert to float and normalize
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply Non-Max Suppression (NMS)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):
            if webcam or source.lower().startswith(('http://', 'https://')):
                p, s, im0, frame = path[i], '', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # Convert to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain whwh
            if len(det):
                # Filter detections based on confidence threshold
                det = det[det[:, 4] >= min_confidence]
                
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if view_img:  # Add bounding box to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference')

            # Display results
            if view_img:
                cv2.imshow("yolov7", im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit when 'q' is pressed
                    cv2.destroyAllWindows()
                    # Release the webcam if used
                    if hasattr(dataset, 'cap'):
                        dataset.cap.release()
                    return

    print(f'Done. ({time.time() - t0:.3f}s)')
    cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed
    
    # Release the webcam if used
    if hasattr(dataset, 'cap'):
        dataset.cap.release()

def main():
    while True:
        # Menu options for the user
        print("###################################################")
        print("Choose what do you want to detect:")
        print("1: Live video detection with the computer camera")
        print("2: Detect on video MP4 saved on this computer")
        print("3: Detect on online video (YouTube)")
        print("4: Exit program")
        print("###################################################")
        n = input("Enter your selection: ")

        if n == '1':
            detect_real_time()
        elif n == '2':
            VPath = input("Enter the Path of the video (Warning use /): ")
            detect_real_time(source=VPath)
        elif n == '3':
            VLink = input("Enter the Link of the YouTube video: ")
            stream_url = get_youtube_stream_url(VLink)
            detect_real_time(source=stream_url)
        elif n == '4':
            print("Bye bye")
            break
        else:
            print("Wrong Number")


################################################################
if __name__ == '__main__':

    main()