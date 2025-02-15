import os
import string
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool, cpu_count
import functools
from tqdm import tqdm

# Define constants
INPUT_FOLDER = "./input"
OUTPUT_FOLDER = "fonts"
IMAGE_SIZE = 64
PADDING = 10

# Set LETTER_MODE to "both", "upper", or "lower"
LETTER_MODE = "upper"

# Precompute font sizes for different text heights
font_sizes_cache = {}

# Preload a large image for measuring font sizes
image_for_measuring = Image.new('RGB', (200, 200))


def get_font_size_for_height(text, target_height, font_path):
    """
    Optimized method to calculate the font size using binary search.
    """
    if (text, target_height, font_path) in font_sizes_cache:
        return font_sizes_cache[(text, target_height, font_path)]

    # Binary search for font size
    low, high = 1, 1000  # Set high as an arbitrary large value for font size
    optimal_font_size = 1
    while low <= high:
        mid = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except Exception:
            return optimal_font_size  # If loading fails, return the previous valid size
        
        draw = ImageDraw.Draw(image_for_measuring)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_height = (bbox[3] - bbox[1]) if bbox is not None else 0
        
        if text_height >= target_height:
            optimal_font_size = mid
            high = mid - 1  # Try smaller sizes to get the exact fit
        else:
            low = mid + 1  # Increase the font size

    font_sizes_cache[(text, target_height, font_path)] = optimal_font_size
    return optimal_font_size


def create_directories():
    for lang in ["base", "final"]:
        lang_path = os.path.join(OUTPUT_FOLDER, lang)
        os.makedirs(lang_path, exist_ok=True)


def get_style_chars(mode):
    """Return the set of style characters based on mode."""
    if mode == "upper":
        return string.ascii_uppercase
    elif mode == "lower":
        return string.ascii_lowercase
    else:  # both
        return string.ascii_uppercase + string.ascii_lowercase

def get_target_chars(mode):
    """Return the set of Armenian characters based on mode.
       Uppercase Armenian: U+0531 to U+0556
       Lowercase Armenian: U+0561 to U+0586
    """
    if mode == "upper":
        return "".join(chr(c) for c in range(0x0531, 0x0556 + 1))
    elif mode == "lower":
        return "".join(chr(c) for c in range(0x0561, 0x0586 + 1))
    else:  # both
        upper = "".join(chr(c) for c in range(0x0531, 0x0556 + 1))
        lower = "".join(chr(c) for c in range(0x0561, 0x0586 + 1))
        return upper + lower



def check_font_support(font_path, mode="both"):
    """
    Check if the font supports all required style and target letters based on the mode.
    Returns True if all letters are supported, else False.
    """
    test_font_size = 50  # Arbitrary test size
    try:
        font = ImageFont.truetype(font_path, test_font_size)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        return False

    def has_glyph(char):
        try:
            dummy_img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), "white")
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), char, font=font)
            if bbox is None:
                return False
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width > 0 and height > 0
        except Exception:
            return False

    style_chars = get_style_chars(mode)
    for char in style_chars:
        if not has_glyph(char):
            return False

    target_chars = get_target_chars(mode)
    for char in target_chars:
        if not has_glyph(char):
            return False

    return True


def save_character_image_to_folder(char, folder, font_path):
    """
    A wrapper for saving character images to a specified folder.
    """
    try:
        img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), "white")
        draw = ImageDraw.Draw(img)

        font_size = get_font_size_for_height(char, IMAGE_SIZE - PADDING * 2, font_path)
        font = ImageFont.truetype(font_path, font_size)

        # Draw text with anchor at the bottom middle.
        draw.text((IMAGE_SIZE / 2, IMAGE_SIZE - PADDING), char, anchor='mb', font=font, fill='black')

        prefix = 'upper' if char.isupper() else 'lower'
        file_path = os.path.join(folder, f"{char}_{prefix}.png")
        img.save(file_path)
    except Exception as e:
        print(f"Error saving {char} from font {font_path}: {e}")


def generate_images_for_font(font_path, mode="both"):
    """
    Generate images for the given font.
    Only process the font if it supports all required characters based on mode.
    """
    font_name = os.path.splitext(os.path.basename(font_path))[0]

    # Skip font if missing any required characters.
    if not check_font_support(font_path, mode):
        print(f"Font '{font_name}' is missing some required glyphs for mode '{mode}'. Skipping.")
        return

    style_folder = os.path.join(OUTPUT_FOLDER, "base", font_name)
    target_folder = os.path.join(OUTPUT_FOLDER, "final", font_name)
    os.makedirs(style_folder, exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)

    style_chars = get_style_chars(mode)
    target_chars = get_target_chars(mode)

    try:
        # Use multiprocessing to generate images in parallel.
        with Pool(processes=cpu_count()) as pool:
            # Generate style images.
            list(pool.imap(
                functools.partial(save_character_image_to_folder, font_path=font_path, folder=style_folder),
                style_chars
            ))
            # Generate target images.
            list(pool.imap(
                functools.partial(save_character_image_to_folder, font_path=font_path, folder=target_folder),
                target_chars
            ))
    except Exception as e:
        print(f"Error processing font '{font_name}': {str(e)}")


if __name__ == "__main__":
    create_directories()

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
    else:
        font_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER)
                      if f.lower().endswith('.ttf')]
        if not font_files:
            print(f"No TTF fonts found in the input folder: {INPUT_FOLDER}")
        else:
            for font_path in tqdm(font_files, desc="Processing Fonts"):
                generate_images_for_font(font_path, mode=LETTER_MODE)
