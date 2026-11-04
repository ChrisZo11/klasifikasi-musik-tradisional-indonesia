import json
import glob
import os
import re
import base64

notebooks = glob.glob('notebooks/*.ipynb')
os.makedirs('assets/notebook_images', exist_ok=True)

extracted_files = []

for nb_path in sorted(notebooks):
    nb_name = os.path.splitext(os.path.basename(nb_path))[0]
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    last_markdown_header = ""
    for cell_idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        source_text = "".join(cell.get('source', []))
        
        if cell_type == 'markdown':
            # find headers like ## Title
            headers = re.findall(r'^#+\s+(.+)$', source_text, re.MULTILINE)
            if headers:
                last_markdown_header = headers[-1].strip()
        elif cell_type == 'code':
            # Check for plot title in code
            plot_titles = re.findall(r'(?:title|suptitle)\s*\(\s*["\']([^"\']+)["\']', source_text)
            savefig_matches = re.findall(r'savefig\s*\(\s*["\']([^"\']+)["\']', source_text)
            
            title_hint = ""
            if savefig_matches:
                title_hint = os.path.splitext(os.path.basename(savefig_matches[0]))[0]
            elif plot_titles:
                title_hint = plot_titles[0]
            elif last_markdown_header:
                title_hint = last_markdown_header
            
            # Clean title hint to be valid filename
            clean_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title_hint).strip('_')
            if len(clean_title) > 50:
                clean_title = clean_title[:50].rstrip('_')
                
            img_in_cell_idx = 0
            for out in cell.get('outputs', []):
                data = out.get('data', {})
                for mime in ['image/png', 'image/jpeg']:
                    if mime in data:
                        raw_data = data[mime]
                        if isinstance(raw_data, list):
                            raw_data = "".join(raw_data)
                        
                        img_bytes = base64.b64decode(raw_data)
                        ext = 'png' if 'png' in mime else 'jpg'
                        
                        if clean_title:
                            filename = f"{nb_name}__cell{cell_idx:02d}_{clean_title}"
                        else:
                            filename = f"{nb_name}__cell{cell_idx:02d}_img{img_in_cell_idx}"
                        
                        if img_in_cell_idx > 0:
                            filename += f"_{img_in_cell_idx}"
                        filename += f".{ext}"
                        
                        out_path = os.path.join('assets/notebook_images', filename)
                        with open(out_path, 'wb') as img_out:
                            img_out.write(img_bytes)
                        
                        extracted_files.append({
                            'notebook': nb_name,
                            'cell': cell_idx,
                            'title_hint': title_hint,
                            'path': out_path,
                            'size_kb': len(img_bytes) / 1024
                        })
                        img_in_cell_idx += 1

print(f"Total extracted: {len(extracted_files)} images\n")
for item in extracted_files:
    print(f"[{item['notebook']}] Cell {item['cell']} -> {item['path']} ({item['size_kb']:.1f} KB) | {item['title_hint']}")
