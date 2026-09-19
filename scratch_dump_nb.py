import json

with open('notebooks/18_exp16_yolo11m_p2_sanity.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch_nb_output.txt', 'w', encoding='utf-8') as f:
    for cell in nb.get('cells', []):
        for out in cell.get('outputs', []):
            if 'text' in out:
                f.write(''.join(out['text']) + '\n')
            elif 'data' in out and 'text/plain' in out['data']:
                f.write(''.join(out['data']['text/plain']) + '\n')
