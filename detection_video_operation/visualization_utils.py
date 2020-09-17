from PIL import ImageDraw ,ImageFont
import numpy as np
import pdb


def show_results(img,age,gender, bounding_boxes, facial_landmarks = []):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    npa = np.asarray(age, dtype=np.float32)
    npa = npa.reshape(npa.shape[0],-1)
    npa = npa[-bounding_boxes.shape[0]:]
    npb = np.asarray(gender, dtype=np.object)
    npb = npb.reshape(npb.shape[0],-1)
    npb = npb[-bounding_boxes.shape[0]:]
 
    bounding_boxes = np.append(bounding_boxes, npa, axis=1)
    bounding_boxes = np.append(bounding_boxes, npb, axis=1)

    for b in bounding_boxes:
        
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline = 'black',width=3)
        draw.text((b[0], b[1]-25), text = ("Age: "+str(b[5])+"\nGender: " +str(b[6])),  align ="left",fill=(0,0,255),width=15)
  
    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline = 'blue',width = 5)
    return img_copy
    
    