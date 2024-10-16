import argparse
import time
from pathlib import Path

import cv2
import torch
from numpy import random
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


def detect_real_time(weights='yolov7.pt', source='0', conf_thres=0.25, iou_thres=0.45, imgsz=980):
    source = source # Source de la webcam
    view_img = True  # Afficher les résultats
    save_txt = False  # Sauvegarder les résultats dans des fichiers .txt
    save_img = False  # Sauvegarder les images avec les détections
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    device = select_device('')  # Utiliser le GPU s'il est disponible, sinon utiliser le CPU
    half = device.type != 'cpu'  # Demi-précision uniquement sur CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # Charger le modèle
    stride = int(model.stride.max())  # Stride du modèle
    imgsz = check_img_size(imgsz, s=stride)  # Vérifier la taille de l'image

    # Set Dataloader
    cudnn.benchmark = True  # True pour accélérer l'inférence avec des images de taille constante
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # Conversion en flottant complet (float)

        img /= 255.0  # Conversion en échelle de gris (0-255 à 0.0-1.0)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # Détecter chaque image
            if webcam:  # Batch size >= 1
                p, s, im0, frame = path[i], '', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # Convertir en chemin
            save_path = str(p.name)  # img.jpg
            txt_path = str('labels/' + p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Gain de normalisation whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Ajouter la boîte à l'image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference')

            # Afficher les résultats
            if view_img:
                cv2.imshow("yolov7", im0)
                if cv2.waitKey(1) == ord('q'):  # Quitter si 'q' est enfoncé
                    break

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    detect_real_time()
