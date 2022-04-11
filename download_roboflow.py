import torch
from roboflow import Roboflow

print('Setup complete. Using torch %s %s' % (
    torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

rf = Roboflow(api_key="AKFMAjaV9dbFzJ9BtBKp")
project = rf.workspace("augmented-startups").project("vehicle-registration-plates-trudk")
dataset = project.version(2).download("yolov5")

# Загрузка данных с Roboflow v1
# rf = Roboflow(api_key="AKFMAjaV9dbFzJ9BtBKp")
# project = rf.workspace("xugar-zurbag").project("plate-snxdn")
# dataset = project.version(4).download("yolov5")

'Обучение Работает'
# python train.py --img 640 --epochs 600 --data Plate-4/data.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --name yolov5s_results  --cache
# Без ограничений при отсутстиви улучшений в обучении
# python train.py --img 640 --patience 0 --epochs 1000 --data Plate-4/data.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --name yolov5s_results  --cache

''

# Ругается на память
# python train.py --img 416 --batch 16 --epochs 100 --data Plate-4\data.yaml  -pip-weights 'yolov5x.pt'

# Какогото лешего не ищет путь
#  python train.py --data Platev4/data.yaml --cfg yolov5s.yaml --weights '' --batch-size 16


# cd yolo5
# python train.py --img 640 --batch 16 --epochs 600 --data models/plate-4/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache

# python train.py --date c:\\plate\data.yaml -cfg yolov5x.yaml --weights '' --batch-size 64

# Если ругнется что нет tensorboard
# conda install -c conda-forge tensorboard
# conda install -c conda-forge/label/cf201901 tensorboard
# conda install -c conda-forge/label/cf202003 tensorboard
