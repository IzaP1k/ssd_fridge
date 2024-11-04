import polars as pl
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageFilter
import os
import cv2 as cv
import csv
from PIL import Image
import os
import torch
import torchvision.transforms as transforms


def sliding_window_for_images(folder_path):
    window_size = (1000, 1000)
    step_size = 500

    for image_name in os.listdir(folder_path):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            user_image = Image.open(image_path)

            gray_image = user_image.convert('L')

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(gray_image, cmap='gray')

            for y in range(0, gray_image.size[1] - window_size[1], step_size):
                for x in range(0, gray_image.size[0] - window_size[0], step_size):
                    rect = patches.Rectangle((x, y), window_size[0], window_size[1], edgecolor='r', facecolor='none',
                                             linewidth=2)
                    ax.add_patch(rect)

            plt.title(f"Sliding Window on Image: {image_name}")
            plt.show()


def use_filter(folder_path, path, show=False):
    for image_name in os.listdir(folder_path):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)

            src = cv.imread(image_path, cv.IMREAD_COLOR)

            src1 = cv.GaussianBlur(src, (5, 5), 0)
            src2 = cv.GaussianBlur(src, (3, 3), 5)

            gray_image1 = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
            gray_image2 = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)

            laplace_filtered_image = cv.Laplacian(gray_image1, ddepth=cv.CV_16S, ksize=5)
            laplace_filtered_image = cv.convertScaleAbs(laplace_filtered_image)  # CV_8U

            new_path = path + r"\laplace"
            laplace_output_path = os.path.join(new_path, f'laplace_{image_name}')
            os.makedirs(new_path, exist_ok=True)
            cv.imwrite(laplace_output_path, laplace_filtered_image)

            sobel_x = cv.Sobel(src=gray_image1, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)
            sobel_y = cv.Sobel(src=gray_image1, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)
            sobel = cv.convertScaleAbs(sobel_x + sobel_y)

            new_path = path + r"\sobel"
            sobel_output_path = os.path.join(new_path, f'sobel_{image_name}')
            os.makedirs(new_path, exist_ok=True)
            cv.imwrite(sobel_output_path, sobel)

            canny_filtered_image = cv.Canny(gray_image2, threshold1=45, threshold2=50, L2gradient=True)

            new_path = path + r"\canny"
            canny_output_path = os.path.join(new_path, f'canny_{image_name}')
            os.makedirs(new_path, exist_ok=True)
            cv.imwrite(canny_output_path, canny_filtered_image)

            if show:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

                ax1.imshow(laplace_filtered_image, cmap='gray')
                ax1.set_title(f"Laplace Filter: {image_name}")

                ax2.imshow(canny_filtered_image, cmap='gray')
                ax2.set_title(f"Carry Filter: {image_name}")

                ax3.imshow(sobel, cmap='gray')
                ax3.set_title(f"Sobel: {image_name}")

                plt.show()


def check_overlap(xmin_obj, xmax_obj, ymin_obj, ymax_obj, window_xmin, window_xmax, window_ymin, window_ymax,
                  show=False):
    if show:
        print(f"Objekt: xmin={xmin_obj}, xmax={xmax_obj}, ymin={ymin_obj}, ymax={ymax_obj}")
        print(f"Okno: xmin={window_xmin}, xmax={window_xmax}, ymin={window_ymin}, ymax={window_ymax}")

    return window_xmax > xmin_obj and window_ymax > ymin_obj


def save_sliding_window_and_annotation(annotation_df, window_size, step_size, image_dir, path, special_name=None):
    output_dir = f"/{special_name}_sliding_windows"
    new_path = path + output_dir
    os.makedirs(new_path, exist_ok=True)

    results = []

    for filename in annotation_df['filename'].unique():

        # Dodanie specjalnej nazwy do pliku, jeśli podano
        if special_name is not None:
            filename = f"{special_name}_" + filename

        image_path = os.path.join(image_dir, filename)
        image = cv.imread(image_path)

        if image is None:
            print(f"Image {filename} not found.")
            continue

        image_height, image_width = image.shape[:2]

        # Filtrowanie dla danego pliku za pomocą Polars
        if special_name is None:
            img_annotations = annotation_df.filter(pl.col('filename') == filename)
        else:
            filename = filename.removeprefix(f"{special_name}_")
            img_annotations = annotation_df.filter(pl.col('filename') == filename)

        for x in range(0, image_width, step_size):
            for y in range(0, image_height, step_size):
                window_xmin, window_xmax = x, min(x + window_size, image_width)
                window_ymin, window_ymax = y, min(y + window_size, image_height)

                window = image[window_ymin:window_ymax, window_xmin:window_xmax]
                window_filename = f"{filename}_{window_xmin}_{window_ymin}.jpg"
                window_path = os.path.join(new_path, window_filename)

                cv.imwrite(window_path, window)
                '''
                cv.imwrite(window_path, window)
                user_image = Image.open(window_path)

                plt.imshow(user_image, cmap='gray')
                plt.show()
                '''

                contains_object = False

                for obj in img_annotations.iter_rows():
                    xmin_obj, ymin_obj, xmax_obj, ymax_obj = obj[1], obj[2], obj[3], obj[4]

                    if check_overlap(xmin_obj, xmax_obj, ymin_obj, ymax_obj, window_xmin, window_xmax, window_ymin,
                                     window_ymax):
                        contains_object = True

                        break

                results.append([window_filename, contains_object])

    # Zapis wyników w pliku CSV
    csv_filename = f"{special_name}_sliding_window.csv" if special_name else "sliding_window.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['window_filename', 'contains_object'])
        writer.writerows(results)


def for_train_valid_test(show):
    folders = ['train', 'valid', 'test']

    for folder in folders:
        df = pl.read_csv(fr"C:\Users\izaol\fridge_food\jedzenie\{folder}\_annotations.csv")

        path = f"./jedzenie/{folder}/edit"

        os.makedirs(path, exist_ok=True)

        use_filter(f"jedzenie/{folder}", path, show=show)

        output_sobel = path + "/sobel"
        output_canny = path + "/canny"
        output_laplace = path + "/laplace"

        save_sliding_window_and_annotation(df, 1000, 500, output_sobel, path, special_name='sobel')
        save_sliding_window_and_annotation(df, 1000, 500, output_canny, path, special_name='canny')
        save_sliding_window_and_annotation(df, 1000, 500, output_laplace, path, special_name='laplace')


def pad_image_to(image, gray, number, new_number = 300):
    width, height = image.size
    if width == number and height == number:
        return image
    if gray:
        image = image.convert("L")
        new_image = Image.new("L", (number, number), 0)
    else:
        new_image = Image.new("RGB", (number, number), (255, 255, 255))
    left = (number - width) // 2
    top = (number - height) // 2
    new_image.paste(image, (left, top))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((new_number, new_number)),
        transforms.ToPILImage()
    ])
    resized_image = transform(new_image)

    return resized_image


def process_images_in_folder(folder_path, number, gray=False):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            if img.size != (number, number):
                padded_img = pad_image_to(img, gray, number)
                padded_img.save(img_path)
                print(f"Zmieniono rozmiar {filename}")
            else:
                print(f"{filename} ma już odpowiedni rozmiar")


def find_max_image_dimensions(folders):
    max_width, max_height = 0, 0
    max_image_path = ""

    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(root, file)
                    with Image.open(image_path) as img:
                        width, height = img.size
                        if width * height > max_width * max_height:
                            max_width, max_height = width, height
                            max_image_path = image_path

    print(f"Max wymiary: {max_width}x{max_height} zdjęcia: {max_image_path}")