## HW 1

### Проблема трех тел
Проблема трех тел — это задача классической механики, где три массивных тела взаимодействуют по закону тяготения Ньютона. 
Решение для движения этих тел невозможно аналитически, но доступно численно через моделирование. 
В романе Лю Цысиня "Проблема трех тел" описывается планета Трисолярис в системе из трех звезд, где нестабильные орбиты угрожают выживанию цивилизации.  
Вы - ученый Трисолярис.  
Ваша задача — помочь жителям Трисоляриса предсказать катастрофы.

### Задача

Смоделировать движение трех звезд не так уж трудно, но для этого надо знать начальные условия:  начальные координаты, скорости и массы звезд.  
Координаты и скорости звезд известны из наблюдений в телескоп. Но массы звезд остаются неизвестными.  
Из записей предыдущей цивилизации известно, что массы связаны пропорциями: 1:4:16, но неизвестно, какая звезда соответствует какой массе.  
Ваша цель — определить массы звезд по наблюдениям их движения за небольшой промежуток времени.

![3b_problem.gif](data%2F3b_problem.gif)

### Данные
Вам представлены данные о движении трех звезд за много много лет, каждый год фиксировались координаты и скорости движения звезд. Данные аккуратно записывались в книгу, на одном листе которой помещались записи ровно за 30 лет.
К сожалению, в ходе инвентаризации архивов листы книги перемешались.   
Вам предстоит всего лишь по 30 наблюдениям научиться определять массы звезд, т.е. номер звезды.  
Данные находятся в файле train.csv: каждая строка содержит 363 значение — координаты и скорости трех звезд за 30 лет, и номера каждой звезды (0, 1 или 2).

Вам нужно написать скрипт на PyTorch с MultiLayerPerceptron, который определяет номер звезды.
Для самостоятельной проверки модели используйте val.csv. а для предсказаний — test.csv (без номеров звезд).  
Сохраните результаты предсказаний в submission.csv.

Колонки фичей: y0_b0_x, y0_b0_y, y0_b0_vx, ..., y29_b2_vy. Где y0 - год, b0 - body номер, x - координата, vx - скорость.  
Колонки order0, order1, order2 - порядок звезды.  
target - колонка order0. Будем определять только одну звезду. Колнки order1, order2 - не используются.  

### Порядок сдачи
1. Клонируете master
2. Создаете ветку hw1_name_surname: git checkout -b hw1_name_surname
3. Создайте папку homeworks/hw1/name_surname
4. В качестве шаблона используйте файл homeworks/hw1/hw1_template.py
5. Поместите в свою папку файлы решения hw1.py, submission.csv
6. Пушите свою ветку на гитхаб: git add ...; git commit ...; git push origin hw1_name_surname 
7. pull request в master, в качестве ревьюера назначаете @danlapko

### Примечание
* Это задача классификации;
* Не забывайте логировать процесс обучения и валидации модели;
* Проверяйте качество модели на валидационной выборке с помощью accuracy_score; accuracy_score на validation должен не ниже 0.55;
* Чтоб увеличить точность, попробуйте создать новые признаки из уже имеющихся;