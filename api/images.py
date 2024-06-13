import requests
import pdfplumber
import os
import fitz  
from tqdm import tqdm
from PIL import Image

def download_pdf(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"PDF downloaded and saved as '{output_path}'")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")

def extract_images_from_pdf(pdf_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            images = page.images
            for image_index, img in enumerate(images):
                # Extract image coordinates and adjust bounding box if necessary
                x0, y0, x1, y1 = img["x0"], img["y0"], img["x1"], img["y1"]
                
                # Ensure coordinates are within the page boundaries
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, page.width)
                y1 = min(y1, page.height)

                if abs(x0-x1) > 200 and abs(y0-y1) > 200:

                    # Crop the image within the bounding box
                    cropped_image = page.within_bbox((x0, y0, x1, y1)).to_image()
                    image_filename = f"page_{page_number + 1}_image_{image_index + 1}.png"
                    image_path = os.path.join(output_folder, image_filename)
                    
                    # Save the image
                    cropped_image.save(image_path)
    
    print(f"Extracted images saved in '{output_folder}'")

# Example usage
pdf_url = "https://content.syndigo.com/asset/e3b0eafc-69ed-488f-aa4c-7a07c40f9045/original.pdf"
pdf_path = "downloaded_pdf_file.pdf"
output_folder = "extracted_images"

download_pdf(pdf_url, pdf_path)
extract_images_from_pdf(pdf_path, output_folder)



# workdir = r"C:\Users\Aditya\OneDrive\Desktop\PatSeer Codes"
# os.makedirs(workdir+'\images',exist_ok=True)

# for each_path in os.listdir(workdir):
#     if each_path.endswith(".pdf"):
#         doc = fitz.Document(os.path.join(workdir, each_path))

#         for i in tqdm(range(len(doc)), desc="pages"):
#             for img in tqdm(doc.get_page_images(i), desc="page_images"):
#                 xref = img[0]
#                 image = doc.extract_image(xref)
#                 pix = fitz.Pixmap(doc, xref)

#                 # Check if the image is CMYK
#                 if pix.colorspace.n == 4:
#                     pil_image = Image.frombytes("CMYK", [pix.width, pix.height], pix.samples)
#                     pil_image = pil_image.convert("RGB")
#                 else:
#                     pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

#                 pil_image.save(os.path.join(workdir,'images', "%s_p%s-%s.png" % (each_path[:-4], i, xref)))

# print("Done!")
