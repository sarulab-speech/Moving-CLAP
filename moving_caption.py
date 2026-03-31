def doa_to_str(doa):
    if doa < -0.6:
        direction = "on the left side"
    elif doa < -0.2:
        direction = "in the front-left"
    elif doa < 0.2:
        direction = "in front"
    elif doa < 0.6:
        direction = "in the front-right"
    else:
        direction = "on the right side"
    return direction

def spatialize_caption_moving(caption, meta, with_spatial=True, spatial_only=False, capitalize=True):
    """Generate caption with spatial information for moving sources."""
    
    caption = caption.strip()
    caption = caption.rstrip('.')
    caption = caption.strip()
    if capitalize:
        caption = caption[0].upper() + caption[1:] if caption else caption

    if not with_spatial:
        return f"{caption}."
    
    if meta.get('augmentation_type') == 'non_spatial':
        return f"{caption}."
    
    # spatial caption
    if spatial_only:
        if meta.get('is_stationary') == False:
            start_zone = meta.get('start_zone', None)
            end_zone = meta.get('end_zone', None)
            return f"{caption} moving from {start_zone} to {end_zone}."
        else:
            static_zone = meta.get('start_zone', None)
            return f"{caption} coming from the {static_zone}."
    
    if meta.get('is_stationary') == False:
        start_zone = meta.get('start_zone', None)
        end_zone = meta.get('end_zone', None)
        return f"{caption} moving from {start_zone} to {end_zone}."
    else:
        doa = meta.get('start_doa', None)
        direction = doa_to_str(doa)
        return f"{caption} {direction}."

def meta_to_caption_moving(meta, with_spatial=True, sort_on_doa=False, spatial_only=False):
    """Generate multi-source caption with spatial information for moving sources."""
    lis = []
    
    if meta.get('mixed', False):
        # mixed source
        source_metas = meta.get('source_metas', [])
        for idx, src_meta in enumerate(source_metas):
            if spatial_only:
                base_text = "A sound" if idx == 0 else "the other sound"
                caption_text = spatialize_caption_moving(base_text, src_meta, with_spatial, spatial_only=True, capitalize=(idx == 0))
            else:
                caption_text = spatialize_caption_moving(src_meta["caption"], src_meta, with_spatial, spatial_only=False, capitalize=(idx == 0))

            caption_text = caption_text.rstrip('.')
            sort_key = src_meta.get('start_doa', 0)
            lis.append((caption_text, sort_key))
    else:
        # single source
        if spatial_only:
            caption_text = spatialize_caption_moving("A sound", meta, with_spatial, spatial_only=True, capitalize=True)
        else:
            caption_text = spatialize_caption_moving(meta["caption"], meta, with_spatial, spatial_only=False, capitalize=True)
        caption_text = caption_text.rstrip('.')
        sort_key = meta.get('start_doa', 0)
        lis.append((caption_text, sort_key))
    
    if sort_on_doa or (meta.get('mixed', False) and len(lis) > 1):
        lis.sort(key=lambda x: x[1])
    
    captions = [x[0] for x in lis]
    
    if len(captions) == 0:
        return ""
    elif len(captions) == 1:
        return f"{captions[0]}."
    elif len(captions) == 2:
        return f"{captions[0]}, and {captions[1]}."