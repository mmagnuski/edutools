import re
import os
import os.path as op
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

import numpy as np
import pandas as pd


depol_dct = {'ę': 'e', 'ó': 'o', 'ń': 'n', 'ą': 'a', 'ł': 'l', 'ż': 'z',
             'ź': 'z', 'ć': 'c', 'ś': 's'}
depol_dct.update({k.upper(): v.upper() for k, v in depol_dct.items()})


def init_server(mail, password):
    '''Initialize email server.
    Requires gmail to have unsafe applications turned on.'''
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(mail, password)
    return server


def send_this_mail(server, to='mmagnuski@swps.edu.pl', subject='Test',
                   main_text='To jest test.'):
    # use MIME
    msg = MIMEMultipart('alternative')
    msg.set_charset('utf8')
    msg['FROM'] = 'mmagnuski@swps.edu.pl'
    msg['Subject'] = Header(subject.encode('utf-8'), 'UTF-8').encode()
    msg['To'] = to
    _attach = MIMEText(main_text.encode('utf-8'), 'plain', 'UTF-8')
    msg.attach(_attach)

    server.sendmail('mmagnuski@swps.edu.pl', to, msg.as_string())


def depol(txt):
    '''Remove polish special characters from a string.'''
    txt_lst = list(txt)
    changed = False
    for ch_idx in range(len(txt_lst)):
        if txt_lst[ch_idx] in depol_dct:
            changed = True
            txt_lst[ch_idx] = depol_dct[txt_lst[ch_idx]]

    if changed:
        txt = ''.join(txt_lst)
    return txt


# - [ ] compare with find_email and put common stuff in sep func
def find_email_df(df_col, imnzw):
    '''Find e-mail adress in dataframe column given student name.

    df_col: pd.Series
        Series with `swps.st.edu.pl` e-mails.
    imnzw: str
        String containing surname or name and surname.

    Returns
    -------
    mail: str
        Student e-mail.
    '''

    imnzw = imnzw.split(' ')
    nzw = imnzw[1] if len(imnzw) > 1 else imnzw[0]
    nzw = depol(nzw.lower())
    df_res_idx = np.where(df_col.str.contains(nzw))[0]

    # TODO - notfound behavior (default: error?)
    if len(df_res_idx) == 0:
        print(imnzw)


def find_email(maile, imnzw):
    '''TODO'''
    if isinstance(imnzw, list):
        assert len(imnzw) == 2
        imie, nazwisko = [depol(x.lower()) for x in imnzw]
        good_lines = np.array([imię in line and nazwisko in line
                               for line in maile])
    else:
        imnzw = depol(imnzw.lower())
        good_lines = np.array([imnzw in line for line in maile])

    good_line = np.where(good_lines)[0]
    msg = "Found more than one mail for given (name, surname) pair."
    assert len(good_line) < 2, msg

    if len(good_line) == 0:
        return None

    line_idx = good_line[0]
    mail_line = maile[line_idx]
    if '<' in mail_line:
        start, stop = mail_line.find('<') + 1, mail_line.find('>')
        return mail_line[start:stop]
    else:
        return mail_line.replace('\n', '')


# TODO: max_pnts float daje np.:
#       12 / 12.0 (100%)
#       a powinno:
#       12 / 12 (100%)
def formatuj_wynik(perc=None, pnts=None, max_pnts=None):
    '''Formatuje wyniki do formy tekstowej do wklejenia do maila.

    Examples
    --------
    formatuj_wynik() == ''
    formatuj_wynik(pnts=np.nan) == 'brak'
    formatuj_wynik(pnts=8, max_pnts=10) == '8 / 10 (80%)'
    formatuj_wynik(perc=8) == '8%'
    formatuj_wynik(perc=10, max_pnts=10) == '1 / 10 (10%)'
    '''
    wynik = perc if perc is not None else pnts if pnts is not None else ''
    if isinstance(wynik, str):
        return wynik

    wynik_txt = ''
    if np.isnan(wynik):
        wynik_txt = 'brak'
        return wynik_txt

    if perc is not None and pnts is None and max_pnts is not None:
        pnts = max_pnts * (perc / 100)

    if pnts is not None:
        # turn 1.0 etc. floats to int
        if isinstance(pnts, float):
            to_int = all([ch == '0' for ch in str(pnts).split('.')[1]])
            if to_int:
                pnts = int(pnts)
        frmt = '{}' if isinstance(pnts, int) else '{:.2f}'
        wynik_txt += frmt.format(pnts)
        if max_pnts is not None:
            wynik_txt += ' / {}'.format(max_pnts)
            if perc is None:
                perc = pnts / max_pnts * 100

    if perc is not None:
        if isinstance(perc, float):
            to_int = all([ch == '0' for ch in str(perc).split('.')[1]])
            if to_int:
                perc = int(perc)
        strrepr = str(perc).split('.')
        if len(strrepr) > 1:
            n_decimals = min(2, len(strrepr[1]))
            fmt = '{:.' + str(n_decimals) + 'f}%'
        else:
            fmt = '{}%'

        perc_str = fmt.format(perc)
        if pnts:
            wynik_txt += ' (' + perc_str + ')'
        else:
            wynik_txt = perc_str
    return wynik_txt


# TODO:
# - [X..] _perc w nazwie kolumny -> procenty (0 - 100%), _prop -> proporcje
#        (przeliczane na procenty), _uwagi -> napis, w innym wypadku punkty
def mailsender(wyniki, message=None, subject=None, server=None,
               email_list=None, dry_run=False):
    '''Send emails with results.

    Parameters
    ----------
    wyniki: pandas.DataFrame
        DataFrame with rows representing students and columns various results.
    message: str
        Message to send to each student. This should be a string with
        ``{column_name}`` matching places where results from specific columns
        should be inserted. These columns have to be present in the ``wyniki``
        dataframe.
    subject: str
        E-mail subject.
    server: SMTP server
        Server to use when sending the emails.
    email_list: list | None
        Additional list of e-mails. Provided when dataframe does not contain
        student emails.
    dry_run: bool
        If ``True`` does not send email but only shows what would be sent to
        three randomly chosen students.
    '''
    if server is None and not dry_run:
        raise RuntimeError('You need to provide an email server.')
    if message is None:
        raise RuntimeError('You need to provide an email message text.')
    if subject is None:
        subject = "Wyniki"

    # check student column
    check_student_columns = ['student', 'imnzw', 'name', 'imie', 'imię']
    student_col = [col for col in check_student_columns
                   if col in wyniki.columns]
    if len(student_col) == 0:
        raise ValueError('Could not find student column, has to be either:'
                         ', '.joint(check_student_columns[:-1]) + ' or '
                         + check_student_columns[-1])
    else:
        # TODO: warn that more student cols than one, using the first one
        # TODO: addtional arg student_column
        student_col = student_col[0]

    # check email list
    if email_list is None:
        email_col = [col for col in ['email', 'mail', 'e-mail']
                    if col in wyniki.columns]
        if len(email_col) == 0:
            raise RuntimeError('You need to prorvide email_list or have '
                               '"mail", "email" or "e-mail" column in the'
                               ' results data frame.')
        else:
            email_col = email_col[0]

    # evaluate column names in message
    pattern = r'\{([a-z_]+)\}'
    use_columns = re.findall(pattern, message)
    could_not_find_columns = [col for col in use_columns
                              if col not in wyniki.columns]
    if len(could_not_find_columns) > 0:
        raise ValueError('Could not find some columns referenced in the messag'
                         'e string: ' + ', '.join(could_not_find_columns))

    # check if maxval row
    maxval_idx = np.where(wyniki.loc[:, student_col] == 'maxval')[0]
    if len(maxval_idx) > 0:
        maxval_idx = maxval_idx[0]
    else:
        maxval_idx = None

    idx_collection = wyniki.index
    if dry_run:
        idx = np.random.randint(0, len(idx_collection), size=3)
        idx_collection = idx_collection[idx]

    for idx in idx_collection:
        # do not send if already sent
        if 'sent' in wyniki.columns:
            sent_val = wyniki.loc[idx, 'sent']
            if not np.isnan(sent_val) and sent_val:
                continue

        # format column values according to message
        format_dict = format_columns_for_message(wyniki, idx, use_columns,
                                                 maxval_idx=maxval_idx)
        this_message = message.format(**format_dict)

        student = wyniki.loc[idx, student_col]
        if email_list is not None:
            imnzw = imnzw.split() if ' ' in student else student
            send_to = find_email(email_list, imnzw)
        else:
            send_to = wyniki.loc[idx, email_col]

        if send_to is None:
            raise ValueError('Could not find email for {}'.format(student))

        if not dry_run:
            send_this_mail(server, to=send_to, subject=subject,
                           main_text=this_message)
            print('Sent email to {}'.format(send_to))
            wyniki.loc[idx, 'sent'] = True
        else:
            mail_msg = 'To: {}\nSubject: {}\nMessage:\n{}'.format(
                send_to, subject, this_message)
            print(mail_msg)
            print('\n\n')


def format_columns_for_message(wyniki, idx, use_columns, maxval_idx=None):
    format_dict = {}
    for col in use_columns:
        value = wyniki.loc[idx, col]
        maxval = None if maxval_idx is None else wyniki.loc[maxval_idx, col]
        if col.endswith('_perc'):
            value_txt = formatuj_wynik(perc=value, max_pnts=maxval)
        else:
            value_txt = formatuj_wynik(pnts=value, max_pnts=maxval)
        format_dict[col] = value_txt
    return format_dict


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


def find_same_notebooks(df, nb_dir):
    n_cells = list()
    for student in df.student:
        nb = read_nb(op.join(nb_dir, student + '.ipynb'))
        n_cells.append(len(nb['cells']))

    n_nb = len(n_cells)
    same_cells = np.zeros((n_nb, n_nb), dtype='bool')
    same_nb = same_cells.copy()

    for s_idx in range(n_nb - 1):
        # find next same num of cells:
        same_numcell = np.where(np.array(n_cells[s_idx + 1:]) == n_cells[s_idx])[0]
        same_numcell += s_idx + 1
        if len(same_numcell) > 0:
            nb1 = read_nb(op.join(nb_dir, df.student[s_idx] + '.ipynb'))

        for s2_idx in same_numcell:
            nb2 = read_nb(op.join(nb_dir, df.student[s2_idx] + '.ipynb'))
            same_cells[s_idx, s2_idx] = compare_cell_code(nb1, nb2).all()
            same_nb[s_idx, s2_idx] = nb1 == nb2

    return same_cells, same_nb


def get_execution_count(nb):
    exec_cnt = [nb['cells'][idx]['execution_count']
                for idx in range(len(nb['cells']))
                if 'execution_count' in nb['cells'][idx]]
    return exec_cnt
