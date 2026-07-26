# ALPR Datasets

This directory contains the datasets used for training and evaluation of the Automatic License Plate Recognition (ALPR) system. The raw data is distributed as `data.zip` and extracted into the `data/` folder.

## Folder Structure

```
5.alpr/
├── data/                          # Extracted dataset directory
│   ├── car_images/                # Car images with plate region annotations
│   │   ├── train/                 # Training split
│   │   ├── test/                  # Test split
│   │   └── validation/            # Validation split
│   ├── plate_images/              # Cropped plate character images
│   │   ├── train/
│   │   ├── test/
│   │   └── validation/
│   └── plate_image_with_dummy/    # Cropped plate images with dummy/augmented samples
│       ├── train/
│       ├── test/
│       └── validation/
├── dataset/                       # Dataset loading code (Python)
├── config.py                      # Project configuration
└── DATASETS.md                    # This file
```

## Dataset Overview

There are **3 datasets** in `data/`:

| Dataset | Purpose | Train | Test | Validation |
|---|---|---|---|---|
| `car_images` | Full car images with plate region bounding boxes | 29,342 | 8,352 | 4,240 |
| `plate_images` | Cropped plate character images (digit & letter classification) | 38,762 | 11,118 | 5,610 |
| `plate_image_with_dummy` | Cropped plate images including dummy/augmented samples | 90,938 | 11,204 | 10,830 |

## How Images Are Stored

Images are stored as **JPEG** (`.jpg`) files directly in the split directories. There is no subfolder hierarchy within each split — all images reside directly under `train/`, `test/`, or `validation/`.

- **car_images**: filenames follow the pattern `day_XXXXX.jpg` or `night (NNN).jpg`
- **plate_images**: filenames follow the pattern `NNNNN.jpg` (zero-padded 5-digit numbers)
- **plate_image_with_dummy**: filenames follow the pattern `NNNNN.jpg` (zero-padded 5-digit numbers)

## How Labels Are Stored

Labels are stored as **XML annotation files** in the same directory as the corresponding images, using the **Pascal VOC** format. Each image has a matching XML file with the same base name (e.g., `00001.jpg` → `00001.xml`).

### XML Structure

Each XML file contains:

```xml
<annotation>
    <filename>image_name.jpg</filename>
    <folder>...</folder>
    <size>
        <width>...</width>
        <height>...</height>
        <depth>3</depth>
    </size>
    <object>
        <name>LABEL</name>
        <bndbox>
            <xmin>...</xmin>
            <ymin>...</ymin>
            <xmax>...</xmax>
            <ymax>...</ymax>
        </bndbox>
    </object>
    <!-- ... more <object> entries ... -->
</annotation>
```

### Label Fields

Each `<object>` element contains:
- **`<name>`** — the character or region label (see below)
- **`<bndbox>`** — bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`) in pixel coordinates relative to the original image

## Label Contents

### `car_images` labels

Used for **plate detection** (localizing the license plate region in a full car image). Labels are:

| Label | Meaning |
|---|---|
| `کل ناحیه پلاک` | Whole plate region (the bounding box enclosing the entire license plate) |
| `0`–`9` | Digits found on the plate |
| `الف`, `ب`, `پ`, `ت`, `ث`, `ج`, `د`, `ز`, `ژ` | Persian letters |
| `س`, `ش`, `ص`, `ط`, `ظ`, `ع`, `ق`, `ل`, `م`, `ن`, `ه‍`, `و`, `ی` | Persian letters |
| `ژ (معلولین و جانبازان)` | Special Persian letter (Zhe with diacritic) |

### `plate_images` labels

Used for **character classification** on cropped plate images. Labels are:

| Label | Meaning |
|---|---|
| `0`–`9` | Digits |
| `الف`, `ب`, `پ`, `ت`, `ث`, `ج`, `د`, `ز`, `ژ` | Persian letters |
| `س`, `ش`, `ص`, `ط`, `ع`, `ق`, `ل`, `م`, `ن`, `ه‍`, `و`, `ی` | Persian letters |
| `ژ (معلولین و جانبازان)` | Special Persian letter |

### `plate_image_with_dummy` labels

Same as `plate_images` but with **additional labels** for augmented/dummy samples:

| Additional Labels | Meaning |
|---|---|
| `D`, `S` | Latin letters (appearing in augmented samples) |
| `تشریفات` | Persian word meaning "ceremonies" (appears in augmented samples) |
| `ف`, `گ`, `ک` | Additional Persian letters not present in `plate_images` |

