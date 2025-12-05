# CNN Image Classification Project

Bu proje CIFAR-10 veya kendi veri setiniz ile CNN tabanlı görüntü sınıflandırma yapar.
Tüm eğitim kodu, confusion matrix üretimi, grafik çizimleri, TFJS export ve örnek veri seti yapısı dahildir.

## Klasör Yapısı
```
cnn_image_classification_full/
│
├── train.py
├── requirements.txt
├── README.md
│
├── scripts/
│   └── export_tfjs.py
│
├── models/
├── results/
└── data/
    ├── train/class1/
    ├── train/class2/
    ├── val/class1/
    └── val/class2/
```

## Kurulum
```
pip install -r requirements.txt
```

## Eğitim
```
python train.py
```

## TFJS Export
```
python scripts/export_tfjs.py
```
