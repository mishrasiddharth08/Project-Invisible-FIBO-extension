"""Reference preparation for native img2img/Inpaint, independent of UI layout."""
from types import SimpleNamespace
from PIL import Image,ImageOps,ImageChops
import numpy as np
import cv2


def prepare_edit(source, mask, size, *, only_masked=False, padding=32, blur_x=0, blur_y=0):
    original=ImageOps.exif_transpose(source).convert('RGBA')
    has_alpha=original.getchannel('A').getextrema()!=(255,255)
    source=Image.new('RGB',original.size,'white')
    source.paste(original,mask=original.getchannel('A'))
    if mask is None:
        if only_masked:
            raise ValueError('Only masked editing requires a painted mask.')
        return SimpleNamespace(reference=source.resize(size,Image.Resampling.LANCZOS),mask=None,overlay=None,paste_to=None,support=None,original=original.resize(size),blend=None,has_alpha=has_alpha)
    support=mask.convert('L').resize(source.size,Image.Resampling.NEAREST).point(lambda x:255 if x>=128 else 0)
    bounds=support.getbbox()
    if bounds is None:
        raise ValueError('Paint an area before requesting a masked FIBO edit.')
    array=np.asarray(support)
    if blur_x: array=cv2.GaussianBlur(array,(0,1),float(blur_x))
    if blur_y: array=cv2.GaussianBlur(array,(1,0),float(blur_y))
    blend=ImageChops.multiply(Image.fromarray(array),support)
    if only_masked:
        pad=max(0,int(padding))
        left,top,right,bottom=bounds
        width,height=right-left+2*pad,bottom-top+2*pad
        ratio=size[0]/size[1]
        width=max(width,height*ratio);height=max(height,width/ratio)
        width=min(source.width,int(np.ceil(width)));height=min(source.height,int(np.ceil(height)))
        left=max(0,min(source.width-width,int((left+right-width)/2)))
        top=max(0,min(source.height-height,int((top+bottom-height)/2)))
        box=(left,top,left+width,top+height)
        ref=source.crop(box).resize(size,Image.Resampling.LANCZOS)
        prepared_mask=support.crop(box).resize(size,Image.Resampling.NEAREST)
        overlay=source.convert('RGBA');overlay.putalpha(ImageOps.invert(blend))
        paste_to=(box[0],box[1],width,height)
    else:
        ref=source.resize(size,Image.Resampling.LANCZOS)
        prepared_mask=support.resize(size,Image.Resampling.NEAREST)
        overlay=ref.convert('RGBA')
        overlay.putalpha(ImageOps.invert(blend.resize(size,Image.Resampling.LANCZOS)))
        paste_to=None
        original=original.resize(size,Image.Resampling.LANCZOS)
        support=support.resize(size,Image.Resampling.NEAREST)
        blend=blend.resize(size,Image.Resampling.LANCZOS)
    return SimpleNamespace(reference=ref,mask=prepared_mask,overlay=overlay,paste_to=paste_to,support=support,original=original,blend=blend,has_alpha=has_alpha)


def restore_masked_alpha(image, prepared):
    """Restore source RGBA exactly outside the painted support after native overlay."""
    if prepared.support is None:
        return image
    if image.size!=prepared.original.size:
        raise ValueError('Native composite dimensions do not match the source.')
    result=image.convert('RGBA')
    alpha=Image.composite(Image.new('L',image.size,255),prepared.original.getchannel('A'),prepared.blend)
    result.putalpha(alpha)
    result=Image.composite(result,prepared.original,prepared.support)
    return result if prepared.has_alpha else result.convert('RGB')
