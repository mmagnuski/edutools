import re
import os
import os.path as op
import json
import numpy as np


def read_nb(nb_file):
    '''Read a notebook file.'''
    with open(nb_file, 'r', encoding='utf-8') as content_file:
        content = content_file.read()
    data = json.loads(content)
    return data


def execute_cell(nb, cell):
    exec(nb["cells"][cell]["source"][0])


# - [ ] TODO: match='exact' vs match='similarity'
def compare_cell_code(nb0, nb1):
    '''Compare code of all code cells between two notebooks.'''
    n_cells = len(nb1['cells'])
    is_exact = np.ones(n_cells, dtype='bool')

    for cell_idx in range(n_cells):
        if nb1['cells'][cell_idx]['cell_type'] == 'code':
            is_exact[cell_idx] = (nb1['cells'][cell_idx]['source']
                                  == nb0['cells'][cell_idx]['source'])
    return is_exact


def full_compare_cells(nb0, nb1):
    '''Comapre every pair of notebook cells.'''
    n_cells0 = len(nb0['cells'])
    n_cells1 = len(nb1['cells'])
    cell_exact = np.zeros((n_cells0, n_cells1), dtype='bool')

    for cell_idx0 in range(n_cells0):
        cell0 = nb0['cells'][cell_idx0]['source']
        for cell_idx1 in range(n_cells1):
            cell1 = nb1['cells'][cell_idx1]['source']
            cell_exact[cell_idx0, cell_idx1] = cell0 == cell1
    return cell_exact


def remove_unrelated_cells(nb0, nb1, start_idx):
    '''Remove additional cells from the second notebook (nb1) that are not
    present in the original notebook (nb0).'''

    isit = list()
    for next_idx in range(start_idx, len(nb1['cells'])):
        the_same = nb0['cells'][start_idx]['source'] == nb1['cells'][next_idx]['source']
        if the_same:
            break

    if the_same:
        steps = next_idx - start_idx
        for xx in range(steps):
            nb1['cells'].pop(start_idx)


def find_same_notebooks(nb_dir, df=None):
    '''Scan notebooks in a directory for same content.'''
    n_cells = list()
    nb_files = [f for f in os.listdir(nb_dir) if f.endswith('.ipynb')]

    for nb_file in nb_files:
        try:
            nb = read_nb(op.join(nb_dir, nb_file))
            n_cells.append(len(nb['cells']))
        except json.JSONDecodeError:
            n_cells.append(0)

    n_nb = len(n_cells)
    same_cells = np.zeros((n_nb, n_nb), dtype='bool')
    same_nb = same_cells.copy()
    same_exec = same_cells.copy()

    for s_idx in range(n_nb - 1):
        # find next same num of cells:
        same_numcell = np.where(np.array(n_cells[s_idx + 1:]) == n_cells[s_idx])[0]
        same_numcell += s_idx + 1
        if len(same_numcell) > 0:
            nb1 = read_nb(op.join(nb_dir, nb_files[s_idx]))
            exec1 = get_execution_count(nb1)

        for s2_idx in same_numcell:
            nb2 = read_nb(op.join(nb_dir, nb_files[s2_idx]))
            exec2 = get_execution_count(nb2)
            same_cells[s_idx, s2_idx] = compare_cell_code(nb1, nb2).all()
            same_nb[s_idx, s2_idx] = nb1 == nb2
            if not all(exec1 == np.arange(1, len(exec1) + 1)):
                same_exec[s_idx, s2_idx] = np.mean(exec1 == exec2)

    return same_cells, same_nb, same_exec


def get_execution_count(nb):
    '''Read execution counters from code cells in a notebook.'''
    exec_cnt = [nb['cells'][idx]['execution_count']
                for idx in range(len(nb['cells']))
                if 'execution_count' in nb['cells'][idx]]
    return exec_cnt


# - [ ] consider moving to emosie?
def get_keras_training(nb):
    '''Extract training history from keras progress text output from a
    notebook.'''
    if not isinstance(nb, dict):
        nb = read_nb(nb)

    texts = _scan_cells_for_keras_output(nb)
    if len(texts) > 0:
        progress = list()
        for text in texts:
            this_progress = _collect_training_output(text)
            progress.append(this_progress)
        if len(progress) == 1:
            progress = progress[0]
        return progress
    else:
        return ValueError('Could not find keras output in given notebook.')


def _scan_cells_for_keras_output(nb):
    '''Find text reporting keras progress.'''
    n_cells = len(nb['cells'])
    texts = list()
    for idx in range(n_cells):
        cell = nb['cells'][idx]
        is_code = cell['cell_type'] == 'code'
        if not is_code:
            continue
        if len(cell['outputs']) == 0:
            continue
        has_stream = ('name' in cell['outputs'][0]
                      and cell['outputs'][0]['name'] == 'stdout'
                      and cell['outputs'][0]['output_type'] == 'stream')
        if has_stream:
            text = cell['outputs'][0]['text']
            has_progress = ['Epoch 1/' in line for line in text]
            if any(has_progress):
                first = np.where(has_progress)[0][0]
                texts.append(text[first:])
    return texts


def _collect_training_output(text):
    '''Collect loss, accuracy, validation loss and validation accuracy from
    the keras training progress text output from a notebook cell.'''

    time_pattern = '\] - ([0-9]+)s '
    epoch_pattern = 'Epoch ([0-9]+)\/([0-9]+)'
    lossacc_pattern = ' loss: ([0-9]+\.[0-9]+) - accuracy: ([01]\.[0-9]+)'
    lossacc_val_pattern = (' val_loss: ([0-9]+\.[0-9]+) - '
                           'val_accuracy: ([01]\.[0-9]+)')
    keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'epoch', 'time']
    progress = {key: list() for key in keys}

    for text_idx in range(0, len(text), 2):
        epo = re.findall(epoch_pattern, text[text_idx])

        if len(epo) > 0:
            epoch_idx, epoch_max = [int(x) for x in epo[0]]
            progress['epoch'].append(epoch_idx)

            # extract training loss and accuracy
            lossacc = re.findall(lossacc_pattern, text[text_idx + 1])
            if len(lossacc) > 0:
                loss, acc = [float(x) for x in lossacc[0]]
                progress['loss'].append(loss)
                progress['accuracy'].append(acc)

            # extract validation loss and accuracy if present
            if 'val_loss' in text[text_idx + 1]:
                lossacc_val = re.findall(lossacc_val_pattern,
                                         text[text_idx + 1])
                val_loss, val_acc = [float(x) for x in lossacc_val[0]]
                progress['val_loss'].append(val_loss)
                progress['val_accuracy'].append(val_acc)

            # try to extract time
            time_val = re.findall(time_pattern, text[text_idx + 1])
            if len(time_val) > 0:
                progress['time'].append(int(time_val[0]))

    return progress


def _group_code_lines(cell_code):
    blank_start = [line.startswith('    ') for line in cell_code]

    line_groups = list()
    has_pack = False
    for idx, val in enumerate(blank_start):
        if val:
            this_pack.append(idx)
        else:
            if has_pack:
                line_groups.append(this_pack)
            this_pack = list()
            this_pack.append(idx)
            has_pack = True

    if len(line_groups) == 0 or not this_pack == line_groups[-1]:
        line_groups.append(this_pack)

    return line_groups


def _find_cells_with_imports(nb):
    cells_with_imports = np.where(
        [any(['import' in line for line in cell['source']])
         and not cell['cell_type'] == 'markdown'
         for cell in nb['cells']]
        )[0]
    return cells_with_imports


def eval_cells(cells, notebook=None):
    '''Evalueate cell contents and return variables in a dictionary.'''

    # run imports if notebook given
    if notebook is not None:
        cells_with_imports = _find_cells_with_imports(notebook)
        for cell_idx in cells_with_imports:
            for line in notebook['cells'][cell_idx]['source']:
                if 'import' in line:
                    exec(line)

    # group code lines in each cell
    all_line_groups = list()
    is_code_cell = list()
    is_deepnote_cell = list()
    for cell in cells:
        is_code_cell.append(cell['cell_type'] == 'code')
        is_deepnote_cell.append('deepnote_cell_type' in cell['metadata'])

        if not is_deepnote_cell[-1]:
            cell_code = cell['source']
            line_groups = _group_code_lines(cell_code)
            all_line_groups.append(line_groups)

    before = list(locals().keys())
    before.append('before')

    # evaluate lines
    for idx, cell in enumerate(cells):
        if is_code_cell[idx]:
            if not is_deepnote_cell[idx]:
                line_groups = all_line_groups[idx]
                for grp in line_groups:
                    start, fin = grp[0], grp[-1] + 1
                    lines = ''.join(cell['source'][start:fin])
                    exec(lines)
            else:
                exec(''.join(cell['source']))

    if not is_deepnote_cell[idx]:
        del start, fin, lines, grp
    del idx, cell

    # find created variables
    locals_after = locals()
    before.append('locals_after')
    use_keys = [key for key in locals_after.keys() if key not in before]

    # place them in a dictionary
    out_dct = {}
    for key in use_keys:
        out_dct[key] = locals_after[key]

    return out_dct


def cell_contains(cells, pattern):
    '''Check which cells contain a regular expression pattern.

    Parameters
    ----------
    cells : list of dicts
        json representation of notebook cells.
    pattern : str
        Regular expression pattern.

    Returns
    -------
    has_txt : list
        List of cell indices containing given pattern.
    '''
    n_cells = len(cells)
    has_txt = list()
    for cell_idx in range(n_cells):
        this_cell = cells[cell_idx]
        for line in this_cell['source']:
            if len(re.findall(pattern, line)) > 0:
                has_txt.append(cell_idx)
                continue
    return has_txt


# - [ ] unify with cell_contains
def find_cell_containing_text(nb1, find_text):
    hastext = list()
    n_cells = len(nb1['cells'])
    for idx in range(n_cells):
        contents = nb1['cells'][idx]['source']

        deepnote_nb = 'deepnote_cell_type' in nb1['cells'][idx]['metadata']
        shortlist = isinstance(contents, list) and len(contents) == 1
        if deepnote_nb:
            if shortlist:
                contents = contents[0].split('\n')
            elif isinstance(contents, str):
                contents = contents.split('\n')

        hasit = any([find_text in line for line in contents])
        hastext.append(hasit)
    return hastext
