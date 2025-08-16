from ultralytics import YOLO
import pandas as pd

def match_weight_and_detect(image_path, real_weight, model, weights_dict):
    """
    Подбор confidence для совпадения веса блюда с допустимой погрешностью.
    """

    Confidence_step = 0.05
    Min_confidence = 0.1
    Max_confidence = 0.99
    Allowed_deviation = 0.1  # 10% погрешность
    Confidence_step_step = 0.005
    Max_iterations = 30  # ограничение итераций

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
    iterations = 0

    while Min_confidence <= confidence <= Max_confidence and Confidence_step > 0.04 and iterations < Max_iterations:
        try:
            results = model.predict(image_path, conf=confidence, verbose=False, deterministic=False)
        except Exception as e:
            print(f"[ERROR] Ошибка предсказания: {e}")
            break

        predicted_weight, class_list = calculate_predicted_weight(results, weights_dict)
        if real_weight == 0:
            print("[ERROR] Реальный вес равен 0 — деление невозможно")
            break
        error = abs(predicted_weight - real_weight) / real_weight

        print(f"[DEBUG] Итерация {iterations+1}: conf={confidence:.3f}, "
              f"pred_weight={predicted_weight:.2f}, error={error:.4f}, classes={class_list}")

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

        # Корректировка confidence
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

        iterations += 1

    return {
        "status": "fail",
        "message": "Не удалось достичь соответствия с весом",
        "predicted_weight": predicted_weight,
        "real_weight": real_weight,
        "error": round(error, 4) if predicted_weight is not None else None,
        "confidence": round(confidence, 3),
        "dishes": class_list if predicted_weight is not None else []
    }


result = match_weight_and_detect(
    image_path=r"C:\Users\Kirill\Desktop\Cutlet\CutletV1\images\20250708_185247.jpg",
    csv_path=r"C:\Users\Kirill\Desktop\Cutlet\Menu.csv",
    model=YOLO(r"C:\Users\Kirill\Desktop\Cutlet\yoloLAST\yoloLAST.pt"),
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

