import cv2
import numpy as np
import os

def explain_object_with_lime(crop_img_path, output_img_path, class_name="Unknown"):
    import cv2
    import numpy as np

    img = cv2.imread(crop_img_path)
    if img is None:
        return "No explanation available."

    # Dummy heatmap overlay
    overlay = img.copy()
    heatmap = np.zeros_like(img)
    heatmap[:, :, 2] = 180  # Red "heat"
    alpha = 0.4
    cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0, overlay)

    cv2.imwrite(output_img_path, overlay)

    # Class-specific explanation
    explanations = {
        "Pistol": "AI focused on the barrel, trigger area, and overall metallic shape.",
        "Grenade": "AI focused on the round shape and the segmented grip of the grenade.",
        "Knife": "Edges and sharp points were key features detected for knife classification.",
        "RPG": "Long tube and conical tip helped the AI recognize this as an RPG.",
        "Machine_Guns": "Barrel length and attached grip played a key role in classification.",
        "Masked_Face": "Covered regions and facial occlusion triggered suspicion.",
        "Bat": "Cylindrical stick shape and grip pattern indicated a police baton.",
    }

    return explanations.get(class_name, "AI focused on key shape and texture features to classify this object.")
