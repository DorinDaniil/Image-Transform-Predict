import os
import argparse
import tensorflow_datasets as tfds
from PIL import Image

def save_domainnet_dataset(output_dir, include_test=True):
    """
    Save the DomainNet dataset as JPG images, organized by domain.
    Optionally includes the test split mixed into the same subfolders.

    Args:
        output_dir: Output directory path where images will be saved.
        include_test: If True, also saves the test split into the same subfolders.
    """
    os.makedirs(output_dir, exist_ok=True)
    domains = ['real', 'painting', 'clipart', 'quickdraw', 'infograph', 'sketch']

    for domain in domains:
        domain_dir = os.path.join(output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)

        ds_train = tfds.load(f'domainnet/{domain}', split='train', shuffle_files=False)
        for i, example in enumerate(tfds.as_numpy(ds_train)):
            img = example['image']
            pil_img = Image.fromarray(img)
            img_path = os.path.join(domain_dir, f'{domain}_train_{i+1:06d}.jpg')
            pil_img.save(img_path, 'JPEG', quality=95)
            if (i + 1) % 1000 == 0:
                print(f"Saved {i + 1} train images for domain: {domain}")

        if include_test:
            ds_test = tfds.load(f'domainnet/{domain}', split='test', shuffle_files=False)
            for i, example in enumerate(tfds.as_numpy(ds_test)):
                img = example['image']
                pil_img = Image.fromarray(img)
                img_path = os.path.join(domain_dir, f'{domain}_test_{i+1:06d}.jpg')
                pil_img.save(img_path, 'JPEG', quality=95)
                if (i + 1) % 1000 == 0:
                    print(f"Saved {i + 1} test images for domain: {domain}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save DomainNet dataset as JPG images.')
    parser.add_argument('--output_dir', type=str,
                        default="/mnt/DATA2/dorin/Image-Transform-Predict/data",
                        help='Output directory path where images will be saved.')
    parser.add_argument('--include_test', default=True,
                        help='If set, also saves the test split into the same subfolders.')
    args = parser.parse_args()
    save_domainnet_dataset(args.output_dir, args.include_test)
    print(f"DomainNet dataset saved to {args.output_dir}")
