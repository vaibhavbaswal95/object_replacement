# Import necessary libraries
import os
import argparse
from functools import partial
import cv2
import requests
from io import BytesIO
from pathlib import Path
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import torch
from torchvision.ops import box_convert

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T
from huggingface_hub import hf_hub_download

from diffusers import StableDiffusionInpaintPipeline

import supervision as sv
import logging


class REPLACEPRODUCT:
    def __init__(self, input_image_path, source_prompt, target_prompt):
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG) 
        # Create a console handler and set the log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Set the desired log level for the console handler
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.debug("Logger initialized")
        
        self.input_image_path = input_image_path
        self.source_prompt = source_prompt
        self.target_prompt = target_prompt
        
        # we could load this from config
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filename = "groundingdino_swint_ogc.pth"
        self.ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"

        # model vars 
        self.image = None
        self.image_source = None
        self.ovd_model = None
        self.inpaint_model = None

        # initializing models
        self.init_ovd_model(self.ckpt_repo_id, self.ckpt_filename, self.ckpt_config_filename)
        self.logger.debug("OVD MODEL LOADED")
        self.init_inpaint_pipe()
        self.logger.debug("INPAINT MODEL LOADED")
        
    def init_ovd_model(self, repo_id, filename, ckpt_config_filename, device='cpu'):
        # loading the open-vocab detection model to identify the object through prompts
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file) 
        self.ovd_model = build_model(args)
        args.device = device 
        
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = self.ovd_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = self.ovd_model.eval()
        return True  
    
    def init_inpaint_pipe(self):
        # loading inpianting model and pushing the graph on cuda device
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        self.inpaint_model  = pipe.to("cuda")
        return True

    def ovd(self, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, image_source):
        boxes, logits, phrases = predict(
        model=self.ovd_model, 
        image=self.image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        return boxes
    
    def inpainting(self, image_source, image_mask, target_prompt):
        image_source = Image.fromarray(self.image_source)
        image_mask = Image.fromarray(image_mask)
        
        #resizing, the cost & quality is also affected by this parameter 
        image_source_for_inpaint = image_source.resize((512, 512))
        image_mask_for_inpaint = image_mask.resize((512, 512))
        
        prompt = "(bright crystal versace eau de toilette, ecommerce product photoshoot), high quality, professional photograph, stunning image, HD resolution), close-up shot, detailed bottle, clear label, sparkling cap, ((soft lighting, white background, minimalistic composition), macro lens, f/2.8 aperture, shallow depth of field)"
        #image and mask_image should be PIL images.
        #The mask structure is white for inpainting and black for keeping as is i.e. replace 0 and 1 other wise
        image_inpainting = self.inpaint_model(prompt=target_prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
        image_inpainted = image_inpainting.resize((image_source.size[0], image_source.size[1]))
        return image_inpainted
    
    def replace_object(self):
        # using them as constants for now
        TEXT_PROMPT = self.source_prompt
        BOX_TRESHOLD = 0.45
        TEXT_TRESHOLD = 0.25
        
        self.image_source, self.image = load_image(self.input_image_path)
        self.logger.debug("IMAGE LOADED")
        boxes = self.ovd(TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, self.image_source)
        self.logger.debug("OVD COMPLETE")
        image_mask = self.generate_masks_with_grounding(self.image_source, boxes)
        self.logger.debug("MASK GENERATED")
        image_inpainted = self.inpainting(self.image_source, image_mask, self.target_prompt)
        self.logger.debug("INPAINTING COMPLETE")
        self.save_output(image_inpainted)
        self.logger.debug("OUTPUT SAVED")
        return image_inpainted
        
    # helper functions
    def generate_masks_with_grounding(self, image_source, boxes):
        h, w, _ = image_source.shape
        boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        mask = np.zeros_like(image_source)
        for box in boxes_xyxy:
            x0, y0, x1, y1 = box
            mask[int(y0):int(y1), int(x0):int(x1), :] = 255
        return mask

    def save_output(self, image_inpainted):
        with image_inpainted as im:
            im.save(r'C:\Users\Skynet\codeshop\image1_app_replaced.png')

# Example usage
if __name__ == "__main__":
    input_image_path = 'C://Users//Skynet//codeshop//image1.png'  # Provide the actual path to your input image
    source_prompt = "a pile of cookie"  # Replace with the desired source object
    target_prompt = "a chocolate pastry"  # Replace with the desired target object

    replacer = REPLACEPRODUCT(input_image_path, source_prompt, target_prompt)
    replacer.replace_object()