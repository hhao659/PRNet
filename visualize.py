import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
 
 
if __name__ == '__main__':
    model = YOLO('yolo11Vis-200.pt') # select your model.pt path
    model.predict(source='VisDrone-Select/9999951_00000_d_0000317.jpg',
                  imgsz=640,
                  project='runs/detect/feature',
                  name='Ti-FPN-Vis',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )
