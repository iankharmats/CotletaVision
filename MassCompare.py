from ultralytics import YOLO
import pandas as pd

def match_weight_and_detect(image_path, csv_path, model_path, real_weight):
    # Настройки
    Confidence_step = 0.05
    Min_confidence = 0.1
    Max_confidence = 0.99
    Allowed_deviation = 0.1  # 10% погрешность
    Confidence_step_step = 0.005

    # Загрузка модели и весов
    try:
        model = YOLO(model_path)
    except Exception as e:
        return {"status": "error", "message": f"Ошибка загрузки модели: {e}"}

    try:
        weights_df = pd.read_csv(csv_path, sep=";")
        weights_dict = dict(zip(weights_df.iloc[:, 0], weights_df.iloc[:, 1]))
    except Exception as e:
        return {"status": "error", "message": f"Ошибка чтения CSV-файла: {e}"}

    def calculate_predicted_weight(results, weights_dict):
        boxes = results[0].boxes
        names = results[0].names
        class_ids = boxes.cls.int().tolist()

        total = 0
        classes = []
        for cls_id in class_ids:
            class_name = names.get(cls_id, None)
            if class_name is None:
                print(f"[WARN] Класс с ID {cls_id} не найден")
                continue
            weight = weights_dict.get(class_name)
            if weight is None:
                print(f"[WARN] Вес для класса '{class_name}' не найден")
                continue
            total += weight
            classes.append(class_name)
        return total, classes

    confidence = 0.5
    predicted_weight = None
    direction = 0

    while Min_confidence <= confidence <= Max_confidence and Confidence_step > 0:
        try:
            results = model.predict(image_path, conf=confidence, verbose=False, deterministic=False)
        except Exception as e:
            return {"status": "error", "message": f"Ошибка предсказания: {e}"}

        predicted_weight, class_list = calculate_predicted_weight(results, weights_dict)
        error = abs(predicted_weight - real_weight) / real_weight

        if error <= Allowed_deviation:
            return {
                "status": "ok",
                "message": "Вес соответствует реальному с допустимой погрешностью",
                "predicted_weight": predicted_weight,
                "real_weight": real_weight,
                "error": round(error, 4),
                "confidence": round(confidence, 3),
                "dishes": class_list
            }

        if real_weight > predicted_weight:
            confidence -= Confidence_step
            if direction == 2:
                Confidence_step -= Confidence_step_step
            direction = -1
        else:
            confidence += Confidence_step
            if direction == -1:
                Confidence_step -= Confidence_step_step
            direction = 2

    return {
        "status": "fail",
        "message": "Не удалось достичь соответствия с весом",
        "predicted_weight": predicted_weight,
        "real_weight": real_weight,
        "error": round(error, 4),
        "confidence": round(confidence, 3)
    }


result = match_weight_and_detect(
    image_path=r"C:\Users\Kirill\Desktop\Cutlet\CutletV1\images\20250708_185247.jpg",
    csv_path=r"C:\Users\Kirill\Desktop\Cutlet\Menu.csv",
    model_path=r"C:\Users\Kirill\Desktop\Cutlet\yoloLAST\yoloLAST.pt",
    real_weight=400.0
)

if result["status"] == "ok":
    print(f"[✔] Распознано успешно: {result['dishes']}")
    print(f"Предсказанный вес: {result['predicted_weight']} г, Весы: {result['real_weight']} г, Погрешность: {result['error']*100:.2f}%")
elif result["status"] == "fail":
    print(f"[⛔] Ошибка соответствия веса: {result['message']}")
    print(f"Предсказанный вес: {result['predicted_weight']} г, Весы: {result['real_weight']} г, Погрешность: {result['error']*100:.2f}%")
else:
    print(f"[‼] Ошибка: {result['message']}")
