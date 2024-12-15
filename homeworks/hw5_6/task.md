## HW 5,6

### Описание
Вы хотите создать свой юмористический журнал, но у вас не хватает финансирования на зарплаты авторам. 
Вы решили в духе последних тенденций автоматизировать написание контента, а конкретно анекдотов.

### Ресурсы
* Датасет посредственных анекдотов: https://huggingface.co/datasets/inkoziev/jokes_dialogues
* Доступ к openai api
* Код проверки моделей: [evaluation_openai.py](evaluation_openai.py)

### Задачи
1 (HW5,HW6):
* Добавить свою какую-нибудь G-Eval метрику в [evaluation_openai.py](evaluation_openai.py) по образу и подобию существующей; 
* Прогнать код эвалюации сравнения двух промптов на 30 семплах на вашей метрике.  В процессе разработки использовать num_samples=10 для экономии бюджета
* Сделать выводы.

2 (HW5):
* Отобрать 50-100 анекдотов. Можно автоматически, можно вручную. Желательно не пошлых, а то модерация openai api может не пропустить.
* Затюнить модель с помощью api openai: https://platform.openai.com/docs/guides/fine-tuning
* Использовать модель gpt-4o-mini
* Сравнить результаты затюненой модели с базовой версией [evaluation_openai.py](evaluation_openai.py). Сделать выводы.
* Залить код тюнинга, модифицированный evaluation code под вашу модель и ссылку на веса;

3 (HW6):
* Тоже самое, что и 2, но затюнить модель с помощью Hugging face. На ваш выбор: DPO, ORPO, PEFT
* Рекомендуется брать небольшие модели, например: meta-llama/Llama-3.2-1B-Instruct или meta-llama/Llama-3.2-3B-Instruct