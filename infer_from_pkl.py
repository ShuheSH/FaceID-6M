import cv2
import torch
import numpy as np
from PIL import Image
import pickle
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

# from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps



import json
import os
from tqdm import tqdm

from insightface.app import FaceAnalysis




def face_extraction(image):
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # image = load_image(image_path)

    # prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = face_info['kps'].tolist()
    face_bbox = face_info['bbox'].tolist()
        
    return face_emb, face_kps




def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":
    # user model

    base_model_path = ".../stable-diffusion-xl-base-1.0"
    face_adapter = ".../checkpoints/ip-adapter.bin"
    controlnet_path = ".../checkpoints/ControlNetModel/"



    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)

    # prompt0 = "jpeg artifacts,asian man,early twenties,casual pose,low quality,blurry,poorly drawn,worst quality,photography,vintage,"
    # prompt1 = "photography aesthetic, elegant portrait photography, young man exuding classical beauty and sophistication, pale skin, delicate features, captivating gaze, slightly parted lips, contemporary fashion fused with vintage elements, naturally-lit setting, warm neutral backdrop, soft, diffused lighting highlighting subject's face, subtle shadowing, clear, high-resolution photo,film grain"
    # prompt2 = "portrait, young asian woman sleek, shoulder-length black hair gazes contemplatively towards camera, dark eyes slightlycast, lips closed neutral expression hints poise introspection, softness skin enhanced,"
    # # prompt2 = "portrait, young asian man, gazes contemplatively towards camera, dark eyes slightlycast, lips closed neutral expression hints poise introspection, softness skin enhanced,"
    # prompt3 = "photography aesthetic,Studio portrait photography, striking sexy female model, serene countenance, youthful appearance, light skin, elegant goth makeup emphasizing sharp eyebrows, pronounced eyelashes, dark lipstick, jewelry showcasing sparkling goth earrings,fashionably styled dark hair, exposing clear skin, backdrop complementing the subject's attire, natural light simulation enhancing soft shadows, subtle retouching ensuring smooth skin textures, harmonious color palette imbued with warm, muted tones,film grain,"
    # # prompt3 = "photography aesthetic,Studio portrait photography, striking male model, serene countenance, youthful appearance, light skin, elegant goth makeup emphasizing sharp eyebrows, pronounced eyelashes, dark lipstick,fashionably styled dark hair, exposing clear skin, backdrop complementing the subject's attire, natural light simulation enhancing soft shadows, subtle retouching ensuring smooth skin textures, harmonious color palette imbued with warm, muted tones,film grain,"
    # prompt4 = "a beautiful woman dressed in casual attire, looking energetic and vibrant, positive and upbeat atmosphere"
    # # prompt4 = "a man dressed in casual attire, looking energetic and vibrant, positive and upbeat atmosphere"
    # prompt5 = "A serene ambience,traditional Chinese aesthetic,a young woman,gentle expression,concentration on instrument,traditional Chinese guzheng,flowing pale blue and white hanfu with delicate floral accents,a backdrop of lush foliage,soft natural lighting,harmonious color palette of cool tones,ancient heritage,cultural reverence,timeless elegance,poised positioning amidst rocks,black hair adorned with classical hairpin,embodiment of classical Chinese music and beauty,tranquility amidst nature,subtlety in details,fine craftsmanship of the guzheng,ethereal atmosphere,cultural homage."
    # # prompt5 = "A serene ambience,traditional Chinese aesthetic,a young man,gentle expression,concentration on instrument,traditional Chinese guzheng,flowing pale blue and white hanfu with delicate floral accents,a backdrop of lush foliage,soft natural lighting,harmonious color palette of cool tones,ancient heritage,cultural reverence,timeless elegance,poised positioning amidst rocks,black hair adorned with classical hairpin,embodiment of classical Chinese music and beauty,tranquility amidst nature,subtlety in details,fine craftsmanship of the guzheng,ethereal atmosphere,cultural homage."


    # prompt0 = "a beautiful girl wearing casual shirt in a garden and smiling"

    # prompt0 = "a beautiful girl wearing casual shirt in a garden"

    prompt0 = "facing one side, wearing red sunglasses, a golden chain, and a green cap"

    # prompt0 = "A man on the red carpet"

    # prompt0 = "a man is standing here, his eyes sharp and full of spirit."

    # prompt0 = "A man holding a  cup of coffee"

    # prompt0 = "A man wearing casual shirt in a garden"

    # prompt0 = "a beautiful woman dressed in casual attire, looking energetic and vibrant, positive and upbeat atmosphere"

    # prompt0 = "side face, wearing red sunglasses, a golden chain, and a green cap"

    # prompt0 = "a man is wearing a sunglasses"

    # prompt0 = "a girl looks very sad"

    # prompt0 = "a man is wearing a mask"


    n_prompt = "ng_deepnegative_v1_75t, (badhandv4:1.2), (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands,((monochrome)), ((grayscale)) watermark, moles, large breast, big breast"



    face_image = load_image("") # image path
    


    face_image = resize_img(face_image, size=(1024, 1024))
    # face_image = resize_img(face_image, size=(512, 512))


    face_emb, face_kps = face_extraction(image=face_image)
    face_kps = draw_kps(face_image, face_kps)
    # prompts = [prompt0, prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7,prompt8,prompt9]
    prompts = [prompt0]


    pipe.set_ip_adapter_scale(0.8)


    print("================")

    inference
    for i in range(1):
        print("-------------")
        image = pipe(
            prompt=prompts[i],
            negative_prompt=n_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioing_scale=0.5,
            num_inference_steps=32,
            guidance_scale=5,
        ).images[0]
        print("+++++++++++++++++")
        ind = len(os.listdir("./results/"))
        # image.save("./results/test_%d.jpg" % (i))
        image.save("./results/test_%d.jpg" % (ind))

