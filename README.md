# boosted-MNIST-classifier
Demonstrates the SOTA approach to training a classifier

Requirements:
keras 2.1.0
tensorflow 1.15.3

Проект демонстрирует новый подход к обучению нейросети. Обычный классификатор MNIST с его помощью достигает рекордных результатов среди классификаторов вообще и лучших - среди простых сетей. Отличие от обычного подхода - постепенное добавление данных в обучающий набор, а также улучшенная функция потерь.

Скрипт train_traidional демонстрирует тренировку сети обычным образом. Обучение идет 3 минуты на GeForce GTX-1080Ti и достигает mAP=0.981333315372467 на тренировочном наборе и mAP=0.9793000221252441 на тестовом
Скрипт train_boosted тренирует модель по-новому. Обучение идет около 3 часов и достигает mAP=1.0 на тренировочном наборе и mAP=0.9898999929428101 на тестовом.

Таким образом, путем 60-кратного увеличения времени достигается значительное улучшение касества распознавания.