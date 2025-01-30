import os
import cv2
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock
import re
import cvzone
# Path to your image file
img_path = "2.jpg"

# Load the image
image = cv2.imread(img_path)

# Check if image loading is successful
if image is None:
    raise ValueError(f"Error: Unable to load image from '{img_path}'")

# Resize the image to the desired dimensions
image_resized = cv2.resize(image, (1020, 600))
image_height, image_width = image_resized.shape[:2]

# Save the resized image (optional, for verification)
cv2.imwrite("resized_1.jpg", image_resized)

# Set up the Google API Key
os.environ["GOOGLE_API_KEY"] = ""

# Initialize the Gemini model
gemini_pro = Gemini(model_name="models/gemini-1.5-flash")

# Create the message with the image
msg = ChatMessage(
    role=MessageRole.USER,
    blocks=[ 
        TextBlock(text="Return bounding boxes in the image in the following format as"
        " a list. \n [ymin, xmin, ymax, xmax, object_name]. need response in text"),
        ImageBlock(path="resized_1.jpg", image_mimetype="image/jpeg"),  # Path to the resized image
    ],
)

# Get the response from the model
response = gemini_pro.chat(messages=[msg])

# Print the response (for debugging purposes)
#print("Response:", response.message.content)
bounding_boxes = re.findall(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\w\s]+)\]', response.message.content)
#print(bounding_boxes)
list1=[]
for box in bounding_boxes:
     parts = box.split(',')
#     print(parts)
     numbers = list(map(int, parts[:-1]))
#     print(numbers)
     label = parts[-1].strip()
#     print(label)
     list1.append((numbers,label))
#print(list1)     
for i ,(numbers,label) in enumerate(list1):
#    print(numbers)
     ymin, xmin, ymax, xmax = numbers
#     # Normalize coordinates using the 1000 scale (as per your request)
     x1 = int(xmin / 1000 * image_width)
     y1 = int(ymin / 1000 * image_height)
     x2 = int(xmax / 1000 * image_width)
     y2 = int(ymax / 1000 * image_height)
     cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
     cvzone.putTextRect(image_resized,f'{label}',(x1,y2 - 10),1,1)

cv2.imshow("Image with Bounding Boxes", image_resized)
cv2.waitKey(0)

