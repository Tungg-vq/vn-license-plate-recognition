from ultralytics import YOLO
import shutil
from IPython.display import FileLink

path_to_best_model = 'result/training/Medium with data augmentation/weights/best.pt' 

model = YOLO(path_to_best_model)
if __name__ == "__main__":



    metrics = model.val(
        data='data/YOLO data/dataset.yaml', 
        split='test',
        project='result/evaluation',
        name='Quantitative_Metrics'
    )

    print(f" mAP50: {metrics.box.map50:.4f}")
    print(f" mAP50-95: {metrics.box.map:.4f}")
    print("graph saved in: result/evaluation/Quantitative_Metrics")


    results = model.predict(
    
    source='data/YOLO data/images/test', 
    conf=0.6,             
    save=True,            
    save_crop=True,       
    project='result/evaluation',
    name='Qualitative_Crops'
    )
