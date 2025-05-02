import os
import string
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

# ------------------------------
# Global Constants & Settings
# ------------------------------

# Dataset generation configuration
INPUT_FOLDER = "./input"         # Folder with font files
INPUT_FOLDER_TEST = "./input-test"
OUTPUT_FOLDER = "fonts"          # Base folder for generated images
IMAGE_SIZE = 64                  # Output image size (width and height)
PADDING = 15                     # Padding from the edge for text
LETTER_MODE = "both"             # "upper", "lower", or "both"
DATASET_MODE = "training"        # Use "training" mode to generate paired images
DATASET_MODE = "test"

# Supported font file extensions
SUPPORTED_EXTENSIONS = ('.ttf', '.otf', '.woff', '.woff2')

# Path for the system/Arial font for content images.
ARIAL_FONT = "arial.ttf"

# Cache for computed font sizes for a given text and target height
font_sizes_cache = {}
# A pre-allocated image used for font size measurement
image_for_measuring = Image.new('RGB', (200, 200))

# ------------------------------
# Utility Functions
# ------------------------------

def get_font_size_for_height(text, target_height, font_path):
    """
    Determine the optimal font size so that the given text's height
    fits into the target height. Uses binary search and caches results.
    """
    key = (text, target_height, font_path)
    if key in font_sizes_cache:
        return font_sizes_cache[key]

    low, high = 1, 1000  # Search range for font sizes
    optimal_font_size = 1
    while low <= high:
        mid = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except Exception:
            return optimal_font_size  # Return last valid size if loading fails
        
        draw = ImageDraw.Draw(image_for_measuring)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_height = (bbox[3] - bbox[1]) if bbox else 0
        if text_height >= target_height:
            optimal_font_size = mid
            high = mid - 1  # try to find a smaller size that still meets target
        else:
            low = mid + 1

    font_sizes_cache[key] = optimal_font_size
    return optimal_font_size

def get_characters(language, letter_mode):
    """
    Return a string of characters for the given language and case mode using explicit character lists.
    """
    language = language.lower()

    if language == "english":
        if letter_mode == "upper":
            return string.ascii_uppercase
        elif letter_mode == "lower":
            return string.ascii_lowercase
        else:
            return string.ascii_letters

    elif language == "armenian":
        upper = "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՈւփՔՕՖ"
        lower = "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ"
        if letter_mode == "upper":
            return upper
        elif letter_mode == "lower":
            return lower
        else:
            return upper + lower

    elif language == "russian":
        upper = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        lower = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        if letter_mode == "upper":
            return upper
        elif letter_mode == "lower":
            return lower
        else:
            return upper + lower

    else:
        raise ValueError(f"Unsupported language: {language}")


def create_output_directories(dataset_mode):
    """
    Creates required output directories based on the dataset mode.
    For training mode we create "base" (for custom font images) and "content"
    (for Arial rendered non-English letter).
    """
    if dataset_mode.lower() == "training":
        for folder in ["base", "content"]:
            os.makedirs(os.path.join(OUTPUT_FOLDER, folder), exist_ok=True)
    else:
        os.makedirs(os.path.join(OUTPUT_FOLDER, "test"), exist_ok=True)

def check_font_support(font_path, chars, test_size=50):
    """
    Check if the font renders each character into a visible glyph (not a box or blank).
    """
    try:
        font = ImageFont.truetype(font_path, test_size)
    except Exception as e:
        print(f"[Error] Failed to load font '{font_path}': {e}")
        return False

    def renders_properly(char):
        try:
            dummy_img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), "white")
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), char, font=font)
            if bbox is None:
                return False
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width > 1 and height > 1
        except Exception:
            return False

    return all(renders_properly(c) for c in chars)

def is_similar_to_missing_glyph(img, font_path, image_size, padding, cache={}):
    """
    Compare rendered character image to a known 'missing glyph' template
    from the same font. Uses bbox-cropping and normalized comparison.
    """
    try:
        key = (font_path, image_size, padding)
        if key not in cache:
            # Render tofu character once per font and cache it
            tofu_img = Image.new("L", (image_size, image_size), "white")
            draw = ImageDraw.Draw(tofu_img)
            font_size = get_font_size_for_height("\uFFFF", image_size - 2 * padding, font_path)
            font = ImageFont.truetype(font_path, font_size)
            draw.text((image_size / 2, image_size - padding), "\uFFFF", anchor='mb', font=font, fill='black')

            bbox = tofu_img.getbbox()
            if bbox:
                tofu_crop = tofu_img.crop(bbox).resize((48, 48))
            else:
                tofu_crop = tofu_img.resize((48, 48))
            cache[key] = tofu_crop
        else:
            tofu_crop = cache[key]

        # Process input image
        img_bbox = img.getbbox()
        if not img_bbox:
            return False
        img_crop = img.crop(img_bbox).resize((48, 48))

        # Convert to arrays
        arr1 = np.array(ImageOps.invert(tofu_crop)).astype(np.float32) / 255
        arr2 = np.array(ImageOps.invert(img_crop)).astype(np.float32) / 255

        diff = np.abs(arr1 - arr2)
        mean_diff = np.mean(diff)

        return mean_diff < 0.025  # highly similar => tofu

    except Exception as e:
        print(f"[Glyph compare error] {e}")
        return False


import numpy as np

def save_character_image_numeric(letter, index, output_folder, font_path, image_size, padding):
    try:
        # Render the actual letter
        img = Image.new("L", (image_size, image_size), "white")
        draw = ImageDraw.Draw(img)

        font_size = get_font_size_for_height(letter, image_size - 2 * padding, font_path)
        font = ImageFont.truetype(font_path, font_size)

        draw.text((image_size / 2, image_size - padding), letter, anchor='mb', font=font, fill='black')

        # Render the known tofu glyph in the same way
        tofu_img = Image.new("L", (image_size, image_size), "white")
        tofu_draw = ImageDraw.Draw(tofu_img)
        tofu_draw.text((image_size / 2, image_size - padding), "\uFFFF", anchor='mb', font=font, fill='black')

        # If the images are exactly the same, it's a tofu/missing glyph
        if list(img.getdata()) == list(tofu_img.getdata()):
            print(f"[Skipped] '{letter}' in font '{os.path.basename(font_path)}' — identical to missing glyph.")
            return

        # Save image if it's valid
        file_name = f"{index}.png"
        img.save(os.path.join(output_folder, file_name))

    except Exception as e:
        print(f"[Error] Saving '{letter}' using font '{font_path}': {e}")




# ------------------------------
# Processing Function for Each Font
# ------------------------------

def process_font(font_path, letter_mode, output_dir, image_size, padding, dataset_mode):
    font_name = os.path.splitext(os.path.basename(font_path))[0]

    # Create output folders
    if dataset_mode.lower() == "training":
        base_folder = os.path.join(output_dir, "base", font_name)
        content_folder = os.path.join(output_dir, "content", font_name)
        os.makedirs(base_folder, exist_ok=True)
        os.makedirs(content_folder, exist_ok=True)
    else:
        base_folder = os.path.join(output_dir, "test", font_name)
        os.makedirs(base_folder, exist_ok=True)
        content_folder = None  # not used in inference

    # --- Save all English letters to base/<font_name>/ ---
    english_chars = get_characters("english", letter_mode)
    for idx, char in enumerate(english_chars):
        save_character_image_numeric(char, idx, base_folder, font_path, image_size, padding)

    # --- Save all supported Armenian + Russian letters to content/<font_name>/ ---
    if dataset_mode.lower() == "training":
        all_foreign_chars = []
        for lang in ["armenian", "russian"]:
            chars = get_characters(lang, letter_mode)
            if check_font_support(font_path, chars):
                all_foreign_chars.extend(chars)

        for idx, char in enumerate(all_foreign_chars):
            save_character_image_numeric(char, idx, content_folder, font_path, image_size, padding)



# ------------------------------
# Font File Helpers & Main Generation Function
# ------------------------------

def get_font_files(input_folder):
    """
    Retrieve a list of font files in the input folder with supported extensions.
    """
    return [os.path.join(input_folder, f) for f in os.listdir(input_folder)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)]

def generate_dataset(input_folder, output_folder, dataset_mode, letter_mode, image_size, padding):
    """
    Main function to generate the dataset.
    
    In training mode:
      - For each font file, generate English letters (custom font) in "base".
      - If available, generate a paired non-English letter:
          * A target image (custom font) saved into "base" with a numeric index after English letters.
          * A content image (Arial) saved into "content".
          
    In inference mode, only English letter images (in "test") are generated.
    """
    create_output_directories(dataset_mode)

    if not os.path.isdir(input_folder):
        print(f"[Error] Input folder not found: {input_folder}")
        return

    font_files = get_font_files(input_folder)
    if not font_files:
        print(f"[Warning] No supported font files found in: {input_folder}")
        return

    # Process each font file while showing progress.
    for font_path in tqdm(font_files, desc=f"Processing fonts ({dataset_mode.title()})"):
        process_font(font_path, letter_mode, output_folder, image_size, padding, dataset_mode)

# ------------------------------
# Entry Point
# ------------------------------

if __name__ == "__main__":
    if DATASET_MODE == 'training':
        generate_dataset(INPUT_FOLDER, OUTPUT_FOLDER, DATASET_MODE, LETTER_MODE, IMAGE_SIZE, PADDING)
    else:
        generate_dataset(INPUT_FOLDER_TEST, OUTPUT_FOLDER, DATASET_MODE, LETTER_MODE, IMAGE_SIZE, PADDING)