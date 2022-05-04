from pdf2image import convert_from_path
import glob
import pytesseract as pt
import cv2
import spacy
import matplotlib.pyplot as plt
from matplotlib import cm
from spacy.matcher import PhraseMatcher
import os
from io import StringIO
import pandas as pd
from collections import Counter
nlp = spacy.load('en_core_web_lg')
from tqdm.auto import tqdm



def convert_pdf_image(pdf_file):       
    '''
    Converts PDF to PNG using pdf2image library
    '''

    images = convert_from_path(pdf_file, poppler_path=r'E:\poppler-0.68.0\bin')
    
    for i in range(len(images)):
        images[i].save(f'Input\\page{i:02}.png', 'PNG')        

    print(f'> Extracted Resume Pages: {len(images)}\n')

    return images


def pdfextract(verbose=True):
    '''
    Extracts text from PNG Images
    '''
    
    pt.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    
    text = []
    
    allfiles = glob.glob(f'Input\\*.png')
    allfiles.sort()
    
    if verbose: print('\033[0;32mAnalyzing Resume Text...\033[0m')
    for x in allfiles:
        image = cv2.imread(x)
        extr = pt.image_to_string(image)
        text.append(extr)
    
    return text



# Function to read resumes from the folder one by one

mypath = 'E:\\JupyterNotebooks\\MSI GP65 Jupyters\\Resume Parser\\Input'
onlyfiles = [os.path.join(mypath, f) for f in os.listdir(
    mypath) if os.path.isfile(os.path.join(mypath, f))]


# function that does phrase matching and builds a candidate profile


def create_profile(file, text):    

    print('> Creating Resume Profile...')

    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()    
        
    # below is the csv where we have all the keywords, you can customize your own
    keyword_dict = pd.read_csv('template.csv')
    stats_words = [nlp(text)
                   for text in keyword_dict['Statistics'].dropna(axis=0)]
    NLP_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis=0)]
    ML_words = [nlp(text)
                for text in keyword_dict['Machine Learning'].dropna(axis=0)]
    DL_words = [nlp(text)
                for text in keyword_dict['Deep Learning'].dropna(axis=0)]
    R_words = [nlp(text) for text in keyword_dict['Management'].dropna(axis=0)]
    python_words = [nlp(text)
                    for text in keyword_dict['Python'].dropna(axis=0)]
    Data_Engineering_words = [
        nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis=0)]
    HR_words = [
        nlp(text) for text in keyword_dict['HR'].dropna(axis=0)]
    Banking_words = [
        nlp(text) for text in keyword_dict['Banking'].dropna(axis=0)]

    
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Stats', None, *stats_words)
    matcher.add('NLP', None, *NLP_words)
    matcher.add('ML', None, *ML_words)
    matcher.add('DL', None, *DL_words)
    matcher.add('Management', None, *R_words)
    matcher.add('Python', None, *python_words)
    matcher.add('DE', None, *Data_Engineering_words)
    matcher.add('HR', None, *HR_words)
    matcher.add('F&B', None, *Banking_words)
    doc = nlp(text)

    d = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        # get the unicode ID, i.e. 'COLOR'
        rule_id = nlp.vocab.strings[match_id]
        span = doc[start: end]  # get the matched slice of the doc
        d.append((rule_id, span.text))
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i, j in Counter(d).items())

    # convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords), names=['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(
        ' ', 1).tolist(), columns=['Subject', 'Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split(
        '(', 1).tolist(), columns=['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'], df2['Keyword'], df2['Count']], axis=1)
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))

    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]

    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()
    # converting str to dataframe
    name3 = pd.read_csv(StringIO(name2), names=['Candidate Name'])

    dataf = pd.concat([name3['Candidate Name'], df3['Subject'],
                      df3['Keyword'], df3['Count']], axis=1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace=True)
    
    
    return dataf



def run_main():
        
    files = glob.glob('*.pdf')
    
    df = pd.DataFrame()
    
    for x in files:
        _images = convert_pdf_image(x)
        text = pdfextract()
        dat = create_profile(x, text)
        df = df.append(dat)
        
    
    
    df = df['Keyword'].groupby(
    [df['Candidate Name'], df['Subject']]).count().unstack()
    df.reset_index(inplace=True)
    df.fillna(0, inplace=True)
    df2 = df.iloc[:, 1:]
    df2.index = df['Candidate Name']
    
    FIGSIZE = 15+(1.1 * len(files))     # Expand figsize as per number of files being processed.
    
    cmap = cm.get_cmap('coolwarm')

    plt.rcParams.update({'font.size': 10})
    ax = df2.plot.barh(cmap=cmap, fontsize=13,
                            legend=True, figsize=(FIGSIZE, 9), stacked=True)
    ax.set_xlabel('Candidate Name', fontsize=20)
    ax.set_ylabel('Relevance', fontsize=20)
    ax.set_title('Resume keywords by category', fontsize=22)

    labels = []
    for j in df2.columns:
        for i in df2.index:
            label = str(j)+": " + str(df2.loc[i][j])
            labels.append(label)

    patches = ax.patches
    for label, rect in zip(labels, patches):
        width = rect.get_width()
        if width > 0:
            x = rect.get_x()
            y = rect.get_y()
            height = rect.get_height()
            ax.text(x + width/1.5, y + height*1.15, label, ha='left', va='center')

    plt.show()
    return df2
    

df2 = run_main()

