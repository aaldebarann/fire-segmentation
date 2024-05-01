# Глубокое обучение в анализе гиперспектральных изображений - сегментация пожаров
## Установка
1. Установите неоходимые зависимости из файла requernmants.txt
```bash
pip install -r requirements.txt
```
2. Скачайте файлы обученных моделей по ссылке [Google Drive](https://drive.google.com/drive/folders/1QcJIFJjenfI00Y8Z7DHgiJ03xUzYY7kA?usp=sharing) и поместите в директорию ***models*** 
## Описание репозитория
Корневая директория сожержит примеры использования разработанного функционала.  
***segmentation.py*** - пример сегментации изображения, полученного со спутникого снимка.
  
Директория ***train*** содержит файлы программы, связанные с обучением моделей. 
Файлы обученных моделей необходимо скачать по ссылке [Google Drive](https://drive.google.com/drive/folders/1QcJIFJjenfI00Y8Z7DHgiJ03xUzYY7kA?usp=sharing) и поместить в директорию ***models***. Для сегментации изображения рекомендуется использовать модель "model-efficientnetb2-adam-001.h5". Информацию о точности этой и других обученных моделей можно найти в директории ***doc***.
  
Директория ***utils*** содержит файлы, реализующие логику получения спутникого изображения, его обработки и вывода.
