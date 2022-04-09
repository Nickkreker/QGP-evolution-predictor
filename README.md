# QGP-evolution-predictor

## Установка
Скачайте репозиторий на локальную машину. Для этого используйте
```
git clone https://github.com/Nickkreker/QGP-evolution-predictor.git
```

Чтобы уcтановить приложение, используйте
```
cd QGP-evolution-predictor
pip install .
```

## Использование
Чтобы считать изначальное состояние (плотность энергии) из файла ```samples/snapshot3.dat``` и вывести предсказанную эволюцию в папку ```out```, запустите
```
qgpepr samples/snapshot3.dat out
```

Чтобы посмотреть, какие есть параметры запуска, используйте
```
qgpepr --help
```
