import cv2
import getopt
import numpy as np
import sys
import yolov5
from PIL import ImageOps, ImageEnhance, Image
from pytesseract import pytesseract

# from vision.vision import detect_license_plate, read_license_plate_tesseract

def read_license_plate(image_path: str) -> str:
    return pytesseract.image_to_string(image_path, config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def detect_license_plate(image_path: str, save_crops=False, save_dir="/tmp/crops") -> Image.Image:
    # load model
    model = yolov5.load('keremberke/yolov5n-license-plate')

    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1  # maximum number of detections per image

    # perform inference
    results = model(image_path, size=640)
    crops = results.crop(save=save_crops, save_dir=save_dir)

    return Image.fromarray(crops[0]['im'][..., ::-1])


def remove_shades(img: Image.Image, threshold: int = 64) -> Image.Image:
	# All pixel values > threshold will be set to 255
	# All pixel values <= threshold will be set to 0
	return img.point(lambda x: 255 if x > threshold else 0, '1')


def remove_dark_frame(img: Image.Image) -> Image.Image:
	# Convert PIL Image to OpenCV format
    open_cv_image = np.array(img.convert('L'))

    # Blur the image to reduce noise
    blurred = cv2.GaussianBlur(open_cv_image, (5, 5), 0)

    # Apply a binary threshold after blurring
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours from the binary image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Assume the largest contour is the frame
    c = max(cnts, key=cv2.contourArea)

    # Find the bounding box coordinates from the contour
    x, y, w, h = cv2.boundingRect(c)

    # Crop the original image using the bounding box coordinates
    cropped = open_cv_image[y:y+h, x:x+w]

    # Convert back to PIL format
    cropped_pil = Image.fromarray(cropped)
    return cropped_pil


def enhance(croped_image: Image.Image) -> Image.Image:
	image = ImageOps.grayscale(croped_image)
	image = ImageEnhance.Contrast(image).enhance(5)
	image = ImageEnhance.Sharpness(image).enhance(5)
	image = ImageOps.scale(image, 3.5)
	image = remove_dark_frame(image)
	image = remove_shades(image)
	return image 
	

def licence_plate_from_img(input_path: str):
	print("reading license plate from image: " + input_path)

	image = cv2.imread(input_path)

	crop = detect_license_plate(image)
	crop.save("./croped.png")

	enhanced_path = "./enhanced.png"
	with enhance(crop) as enhanced:
		enhanced.save(enhanced_path)	

	print("license plate: " + read_license_plate(enhanced_path))


if __name__ == "__main__":
	opts, args = getopt.getopt(sys.argv[1:], "hi:", ["input="])
	input_path = ""
	for opt, arg in opts:
		if opt == '-h':
			print('detect.py -i input_image_path')
			sys.exit()
		elif opt in ("-i", "--input"):
			input_path = arg

	licence_plate_from_img(input_path)

