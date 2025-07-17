import os
import shutil

def copy_jpg_to_txt_dir(jpg_dir, txt_dir, output_dir):
    """
    Проверяет имена .jpg файлов в jpg_dir.
    Если в txt_dir есть .txt с таким же именем, копирует .jpg в output_dir,
    откуда удобно перенести в дататсет
    """
    os.makedirs(output_dir, exist_ok=True)
    
    jpg_files = [f.split('.')[0] for f in os.listdir(jpg_dir) if f.endswith('.jpg')]    
    txt_files = [f.split('.')[0] for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for jpg_base in jpg_files:
        if jpg_base in txt_files:
            src_jpg = os.path.join(jpg_dir, f"{jpg_base}.jpg")
            dst_jpg = os.path.join(output_dir, f"{jpg_base}.jpg")

            shutil.copy2(src_jpg, dst_jpg)
            print(f"Скопирован: {src_jpg} -> {dst_jpg}")

if __name__ == "__main__":
    jpg_directory = r"C:\Users\user\Desktop\Акселератор\cotletaVision_data\stepa1+stepa2"
    txt_directory = r"C:\Users\user\Desktop\Python\accelerator2025\main_project_data\obj_train_data"
    output_directory = r"C:\Users\user\Desktop\Акселератор\cotletaVision_data\stepa2_sorted"

    copy_jpg_to_txt_dir(jpg_directory, txt_directory, output_directory)