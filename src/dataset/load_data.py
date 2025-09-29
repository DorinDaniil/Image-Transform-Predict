import os
import tensorflow_datasets as tfds
from PIL import Image

def save_domainnet_dataset(output_dir):
    """
    Save the full DomainNet dataset as JPG images, organized by domain.

    Args:
        output_dir: Output directory path where images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    domains = ['real', 'painting', 'clipart', 'quickdraw', 'infograph', 'sketch']

    for domain in domains:
        domain_dir = os.path.join(output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)

        # Загружаем весь датасет для текущего домена
        ds = tfds.load(f'domainnet/{domain}', split='train', shuffle_files=False)

        for i, example in enumerate(tfds.as_numpy(ds)):
            img = example['image']
            pil_img = Image.fromarray(img)
            img_path = os.path.join(domain_dir, f'{domain}_image_{i+1:06d}.jpg')
            pil_img.save(img_path, 'JPEG', quality=95)

            # Выводим прогресс каждые 1000 изображений
            if (i + 1) % 1000 == 0:
                print(f"Saved {i + 1} images for domain: {domain}")

if __name__ == "__main__":
    output_directory = "/mnt/DATA2/dorin/Image-Transform-Predict/data"
    save_domainnet_dataset(output_directory)
    print(f"Full DomainNet dataset saved to {output_directory}")
