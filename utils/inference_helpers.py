
import cv2
import uuid


def extract_patches(original_image, mask,xscale,yscale):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to store the patches
    patches = []
    
    # Iterate through each contour
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        ox=int(x*xscale)
        oy=int(y*yscale)
        ow=int(w*xscale)
        oh=int(h*yscale)
        

        
        # Extract the patch from the original image
        patch = original_image[oy-5:oy+oh+5, ox-5:ox+ow+5]

        
        
        # Append the patch to the list
        patches.append(patch)
    
    return patches

def extract_patches_ocr(original_image, mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to store the patches
    patches = []
    
    # Iterate through each contour
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        

        
        # Extract the patch from the original image
        patch = original_image[y-5:y+h+5, x-5:x+w+5]

        
        
        # Append the patch to the list
        patches.append(patch)
    
    return patches


def store_crops(original_image,mask,x_scale,y_scale):
    extracted_patches = extract_patches(original_image=original_image,mask=mask,xscale=x_scale,yscale=y_scale)
    cv2.imwrite("patch.jpg",extracted_patches[0])

    return "patch.jpg"


def store_crops_ocr(original_image,mask,path):
    extracted_patches = extract_patches_ocr(original_image=original_image,mask=mask)
    cv2.imwrite(path,extracted_patches[0])

    

