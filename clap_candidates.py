from moving_caption import meta_to_caption_moving

def generate_all_spatial_candidates(caption=None, spatial_only=False):
    """
    create text candidates for single source
    """
    doa_zones = {
        "left side": (-1.0, -0.6),
        "front-left": (-0.6, -0.2),
        "front": (-0.2, 0.2),
        "front-right": (0.2, 0.6),
        "right side": (0.6, 1.0)
    }
    
    candidates = []
    
    for start_zone_name, start_zone_range in doa_zones.items():
        for end_zone_name, end_zone_range in doa_zones.items():
            start_doa = (start_zone_range[0] + start_zone_range[1]) / 2
            end_doa = (end_zone_range[0] + end_zone_range[1]) / 2
            
            if start_zone_name == end_zone_name:
                is_stationary = True
            else:
                is_stationary = False
            
            meta = {
                'caption': caption if caption else '',
                'start_zone': start_zone_name,
                'end_zone': end_zone_name,
                'start_doa': start_doa,
                'end_doa': end_doa,
                'doa': start_doa,
                'is_stationary': is_stationary,
                'mixed': False
            }
            
            full_caption = meta_to_caption_moving(meta, spatial_only=spatial_only)
            
            candidate = {
                'caption': full_caption,
                'start_zone': start_zone_name,
                'end_zone': end_zone_name,
                'start_doa': start_doa,
                'end_doa': end_doa,
                'is_stationary': is_stationary,
                'meta': meta
            }
            
            candidates.append(candidate)
    
    return candidates

def find_best_matching_candidate(meta, candidates):
    true_start = meta.get('start_zone')
    true_end = meta.get('end_zone')
    
    if true_start and true_end:
        for idx, cand in enumerate(candidates):
            if cand['start_zone'] == true_start and cand['end_zone'] == true_end:
                return idx
    return None

def generate_mixed_spatial_candidates(caption1=None, caption2=None, test_order_robustness=False, spatial_only=False, non_mov_mov=True):
    """
    create text candidates for mixed two sources
    """
    doa_zones = {
        "left side": (-1.0, -0.6),
        "front-left": (-0.6, -0.2),
        "front": (-0.2, 0.2),
        "front-right": (0.2, 0.6),
        "right side": (0.6, 1.0)
    }
    
    candidates = []
    
    for s1_start_zone_name, s1_start_zone_range in doa_zones.items():
        for s1_end_zone_name, s1_end_zone_range in doa_zones.items():
            s1_start_doa = (s1_start_zone_range[0] + s1_start_zone_range[1]) / 2
            s1_end_doa = (s1_end_zone_range[0] + s1_end_zone_range[1]) / 2
            s1_is_stationary = (s1_start_zone_name == s1_end_zone_name)
            
            s1_meta = {
                'caption': caption1,
                'start_zone': s1_start_zone_name,
                'end_zone': s1_end_zone_name,
                'start_doa': s1_start_doa,
                'end_doa': s1_end_doa,
                'is_stationary': s1_is_stationary
            }
            
            for s2_start_zone_name, s2_start_zone_range in doa_zones.items():
                if s1_start_zone_name == s2_start_zone_name:
                    continue
                
                for s2_end_zone_name, s2_end_zone_range in doa_zones.items():
                    s2_start_doa = (s2_start_zone_range[0] + s2_start_zone_range[1]) / 2
                    s2_end_doa = (s2_end_zone_range[0] + s2_end_zone_range[1]) / 2
                    s2_is_stationary = (s2_start_zone_name == s2_end_zone_name)
                    if non_mov_mov:
                        if (not s1_is_stationary and not s2_is_stationary):
                            continue
                    if spatial_only:
                        if s1_start_doa > s2_start_doa:
                            # Avoid duplicates by enforcing an order based on DOA
                            continue
                        s2_meta = {
                            'caption': caption2,
                            'start_zone': s2_start_zone_name,
                            'end_zone': s2_end_zone_name,
                            'start_doa': s2_start_doa,
                            'end_doa': s2_end_doa,
                            'is_stationary': s2_is_stationary
                        }
                        mixed_meta = {
                            'mixed': True,
                            'source_metas': [s1_meta, s2_meta],
                            'n_sources': 2
                        }
                        caption = meta_to_caption_moving(mixed_meta, spatial_only=True)
                        candidates.append({
                            'caption': caption,
                            'source1_start_zone': s1_start_zone_name,
                            'source1_end_zone': s1_end_zone_name,
                            'source1_start_doa': s1_start_doa,
                            'source1_end_doa': s1_end_doa,
                            'source2_start_zone': s2_start_zone_name,
                            'source2_end_zone': s2_end_zone_name,
                            'source2_start_doa': s2_start_doa,
                            'source2_end_doa': s2_end_doa,
                            'is_swapped': False
                        })
                    else:
                        s2_meta = {
                            'caption': caption2,
                            'start_zone': s2_start_zone_name,
                            'end_zone': s2_end_zone_name,
                            'start_doa': s2_start_doa,
                            'end_doa': s2_end_doa,
                            'is_stationary': s2_is_stationary
                        }
                        mixed_meta = {
                            'mixed': True,
                            'source_metas': [s1_meta, s2_meta],
                            'n_sources': 2
                        }
                        caption = meta_to_caption_moving(mixed_meta)
                        
                        candidates.append({
                            'caption': caption,
                            'source1_start_zone': s1_start_zone_name,
                            'source1_end_zone': s1_end_zone_name,
                            'source1_start_doa': s1_start_doa,
                            'source1_end_doa': s1_end_doa,
                            'source2_start_zone': s2_start_zone_name,
                            'source2_end_zone': s2_end_zone_name,
                            'source2_start_doa': s2_start_doa,
                            'source2_end_doa': s2_end_doa,
                            'is_swapped': False
                        })
    
    # if need to test order robustness, create swapped candidates
    if test_order_robustness:
        swapped_candidates = []
        for cand in candidates:
            # change to [source2, source1]
            s1_is_stationary = (cand['source1_start_zone'] == cand['source1_end_zone'])
            s1_meta = {
                'caption': caption1,
                'start_zone': cand['source1_start_zone'],
                'end_zone': cand['source1_end_zone'],
                'start_doa': cand['source1_start_doa'],
                'end_doa': cand['source1_end_doa'],
                'is_stationary': s1_is_stationary
            }
            
            s2_is_stationary = (cand['source2_start_zone'] == cand['source2_end_zone'])
            s2_meta = {
                'caption': caption2,
                'start_zone': cand['source2_start_zone'],
                'end_zone': cand['source2_end_zone'],
                'start_doa': cand['source2_start_doa'],
                'end_doa': cand['source2_end_doa'],
                'is_stationary': s2_is_stationary
            }
            mixed_meta = {
                'mixed': True,
                'source_metas': [s2_meta, s1_meta],
            }
            swapped_caption = meta_to_caption_moving(mixed_meta)
            
            swapped_candidates.append({
                'caption': swapped_caption,
                'source1_start_zone': cand['source2_start_zone'],
                'source1_end_zone': cand['source2_end_zone'],
                'source1_start_doa': cand['source2_start_doa'],
                'source1_end_doa': cand['source2_end_doa'],
                'source2_start_zone': cand['source1_start_zone'],
                'source2_end_zone': cand['source1_end_zone'],
                'source2_start_doa': cand['source1_start_doa'],
                'source2_end_doa': cand['source1_end_doa'],
                'is_swapped': True
            })
        
        candidates.extend(swapped_candidates)
    
    return candidates

def find_best_matching_mixed_candidate(meta, candidates, spatial_only=False):
    source_metas = meta.get('source_metas', [])
    assert len(source_metas) == 2, "source_metas should contain exactly 2 sources."

    s1_start = source_metas[0].get('start_zone')
    s1_end = source_metas[0].get('end_zone')
    s2_start = source_metas[1].get('start_zone')
    s2_end = source_metas[1].get('end_zone')
    
    # if zone information is available, look for an exact match
    if all([s1_start, s1_end, s2_start, s2_end]):
        correct_idx = None
        swapped_correct_idx = None
        
        for idx, cand in enumerate(candidates):
            if spatial_only:
                if ((cand['source1_start_zone'] == s1_start and
                    cand['source1_end_zone'] == s1_end and
                    cand['source2_start_zone'] == s2_start and 
                    cand['source2_end_zone'] == s2_end) or 
                    (cand['source1_start_zone'] == s2_start and
                    cand['source1_end_zone'] == s2_end and
                    cand['source2_start_zone'] == s1_start and
                    cand['source2_end_zone'] == s1_end)):
                    correct_idx = idx
            else:
                if not cand.get('is_swapped', False):
                    if (cand['source1_start_zone'] == s1_start and 
                        cand['source1_end_zone'] == s1_end and
                        cand['source2_start_zone'] == s2_start and 
                        cand['source2_end_zone'] == s2_end):
                        correct_idx = idx
            
                elif cand.get('is_swapped', False):
                    if (cand['source1_start_zone'] == s2_start and 
                        cand['source1_end_zone'] == s2_end and
                        cand['source2_start_zone'] == s1_start and 
                        cand['source2_end_zone'] == s1_end):
                        swapped_correct_idx = idx
        
        if correct_idx is not None:
            return correct_idx, swapped_correct_idx
            
    return None, None