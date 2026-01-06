import warnings
from ultralytics import RTDETR

def main():
    warnings.filterwarnings('ignore')

    model = RTDETR('./ultralytics/cfg/models/rt-detr/rtdetr-l-WT-AIFI.yaml')

    data_yaml = './yamls/alldata.yaml'
    imgsz = 640
    epochs = 100
    batch_size = 16
    workers = 1
    device = '1'  # Use GPU 0 for training
    lr0 = 0.0001
    weight_decay = 0.0005
    augment = True

    # model.load('best.pt')
    model.train(data=data_yaml,
                cache=True,
                imgsz=imgsz,
                epochs=epochs,
                batch=batch_size,
                workers=workers,
                device=device,
                lr0=lr0,
                weight_decay=weight_decay,
                augment=augment,
                project = 'runs/train',
                name='WF-DETR',
                patience=0,
                )

if __name__ == '__main__':
    main()
