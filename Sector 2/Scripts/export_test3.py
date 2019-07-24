from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, Frame, FrameBreak, PageTemplate, NextPageTemplate, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle



import os

from PIL import Image as pilImage
import numpy as np

pdfmetrics.registerFont(TTFont('Open Sans', 'Vera.ttf'))

PAGE_HEIGHT=defaultPageSize[1]; PAGE_WIDTH=defaultPageSize[0]
styles = getSampleStyleSheet()



doctitle = 'Test 2IDD Report'
pageinfo = "2-ID-D, July 2019"
titlefont = 'Open Sans'
titlefontsize = 16
bodyfont = 'Open Sans'
bodyfontsize = 9

def FirstPage(canvas, doc):
    canvas.saveState()
    frg_logo = 'C:\\Users\\rishi\\OneDrive\\Documents\\GitHub\\APS\\Sector 2\\Scripts\\Untitled.png'
    im = pilImage.open(frg_logo)
    imwidth, imheight = im.size
    logowidth = doc.width * 0.9
    logoheight = imheight * logowidth / imwidth

    canvas.drawImage(frg_logo, doc.leftMargin*2, doc.bottomMargin*3.5, logowidth, logoheight)

    # canvas.drawString(inch, 0.75 * inch, "First Page / %s" % pageinfo)
    canvas.restoreState()

def myLaterPages(canvas, doc):
    canvas.saveState()
    # canvas.setFont('Times-Roman',bodyfontsize)
    canvas.drawString(doc.leftMargin/2, doc.bottomMargin/2, "Page %d" % (doc.page))
    canvas.drawRightString(PAGE_WIDTH - doc.leftMargin/2, doc.bottomMargin/2, pageinfo)
    canvas.restoreState()

def ScaledImage(filepath, dimension, size):
    im = pilImage.open(filepath)
    imwidth, imheight = im.size

    if dimension == 'width' or 'Width':
        printwidth = size
        printheight = size * imheight/imwidth
    elif dimension == 'height' or 'Height':
        printwidth = size * imwidth/imheight 
        printheight = size
    else:
        return #add error message here later

    image = Image(filepath, 
        width = printwidth,
        height = printheight,
        hAlign = 'CENTER')
    return image

def generate_image_matrix(image_paths, max_num_cols, max_width, max_height):
    margin = 1.01

    x = np.sqrt(len(image_paths))

    if x == np.floor(x):
        guess_num_cols = x
    else:       
        guess_num_cols = np.floor(x) + 1

    num_cols = int(guess_num_cols)
    num_rows = np.ceil(len(image_paths) / num_cols)

    # if max_width/num_cols < max_height/num_rows:
    #     limiting_dimension = 'width'
    #     limiting_size = (max_width/num_cols) / margin
    # else:
    #     limiting_dimension = 'height'
    #     limiting_size = (max_height/num_rows) / margin       

    limiting_size = max_width / num_cols

    if max_height/num_rows < limiting_size:
        limiting_dimension = 'height'
        limiting_size = max_height / num_rows
    else:
        limiting_dimension = 'width'
    limiting_size = limiting_size / margin


    print(limiting_dimension)
    filling = True
    img_matrix = []
    image_index = 0
    while filling:
        row_matrix = []
        for col in range(num_cols):
            if image_index < len(image_paths):
                im = ScaledImage(image_paths[image_index], dimension = limiting_dimension, size = limiting_size)
                row_matrix.append(im)
                image_index += 1
            else:
                filling = False
                row_matrix.append('')
        img_matrix.append(row_matrix)

    imtable = Table(img_matrix,
     colWidths = limiting_size * margin,
     rowHeights = limiting_size * margin)

    return imtable

def read_scan_params(filepath):
    import json

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data

def build_scan_page(doc, Story, scan_number, title, text, overviewmap_image_filepath, scan_params, scan_image_filepaths):
    ### text section
    headerstr = 'Scan ' + str(scan_number) + ': ' + title

    subheaderstr = []
    subheaderstr.append('\tArea: {0:d} x {1:d} um'.format(scan_params['x_range'], scan_params['y_range']))
    subheaderstr.append('\tStep Size: {0:s} um'.format(scan_params['stepsize']))
    subheaderstr.append('\tDwell Time: 50 ms')
    subheaderstr.append('\tEnergy: 16.3 keV')


    Story.append(Paragraph(headerstr, styles['Heading1']))
    for each in subheaderstr:
        Story.append(Paragraph(each, styles['Heading3']))  
    Story.append(Paragraph(text, styles['Normal']))
    # Story.append(PageBreak())
    Story.append(FrameBreak())
    ### overview map section

    imoverview = ScaledImage(overviewmap_image_filepath, 'width', doc.width * 0.45)
    Story.append(imoverview)
    Story.append(FrameBreak())
    # Story.append(PageBreak())

    ### xrf maps section
    imtable = generate_image_matrix(scan_image_filepaths,
        max_num_cols = 4,
        max_width = doc.width,
        max_height = doc.height * 0.6)
    Story.append(imtable)
    Story.append(FrameBreak())
    # Story.append(PageBreak())

    return Story

def build_title_page(doc, Story, title, subtitle):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', fontSize = 24, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='CenterSubtitle', fontSize = 16, alignment=TA_CENTER))

    Story.append(Paragraph(title, styles['CenterTitle']))
    Story.append(FrameBreak())
    Story.append(Paragraph(subtitle, styles['CenterSubtitle']))
    Story.append(Paragraph(datetime.today().strftime('%Y-%m-%d'), styles['CenterSubtitle']))
    Story.append(NextPageTemplate('ScanPage'))
    Story.append(FrameBreak())
    # Story.append(PageBreak())
    return(Story)

def go(doc, Story, outputpath):
    ## title page template

    titleframe = Frame(
        x1 = doc.leftMargin,
        y1 = doc.height/2,
        width = doc.width,
        height = doc.height*.4,
        )

    subtitleframe = Frame(
        x1 = doc.leftMargin,
        y1 = doc.height/2 -doc.topMargin,
        width = doc.width,
        height = doc.height * 0.4,
        )

    ## scan page template
    text_width = doc.width * 0.5
    text_height = doc.height * 0.35

    textframe = Frame(
        x1 = doc.leftMargin,
        y1 = PAGE_HEIGHT - doc.topMargin - text_height, 
        width = text_width,
        height = text_height,
        id = 'textframe')
    overviewmapframe = Frame(
        x1 = doc.leftMargin + doc.width * 0.5,
        y1 = PAGE_HEIGHT - doc.topMargin - text_height, 
        width = doc.width - text_width,
        height = text_height,
        id = 'overviewmapframe')
    xrfframe = Frame(
        x1 = doc.leftMargin,
        y1 = doc.bottomMargin, 
        width = doc.width,
        height = doc.height - text_height,
        id = 'xrfframe')

    doc.addPageTemplates([PageTemplate(id='TitlePage',frames=[titleframe,subtitleframe], onPage = FirstPage),
                         PageTemplate(id='ScanPage',frames=[textframe,overviewmapframe,xrfframe], onPage = myLaterPages)]
                        )
    doc.build(Story)

def report(outputpath, inputpath):
    def get_channels(channels):
        from tkinter import Tk, IntVar, Checkbutton, Button, mainloop, W
        master = Tk()
        
        max_rows = 5

        var = []
        for i, channel in enumerate(channels):
            colnum = int(np.max([0,np.floor((i-1)/max_rows)]))
            rownum = int(np.mod(i, max_rows))

            var.append(IntVar())
            Checkbutton(master, text=channel, variable=var[i]).grid(row=rownum, column = colnum, sticky=W)

        Button(master, text = 'Build Report', command = master.quit).grid(row = 6, sticky = W)
        mainloop()

        selected_channels = [channels[x] for x in range(len(channels)) if var[x].get() == 1]

        return selected_channels


    scans = os.listdir(inputpath)
    scan_nums = [x[5:9] for x in scans]
    scans = [x for _,x in sorted(zip(scan_nums,scans))]

    all_channels = os.listdir(os.path.join(inputpath, scans[0], 'images'))
    channels = get_channels(all_channels)


    doc = SimpleDocTemplate(outputpath,
        showBoundary = 0)
    Story = []
    # Story.append(NextPageTemplate('TitlePage'))

    Story = build_title_page(
        doc = doc,
        Story = Story,
        title = 'TITLE STRING GOES HERE',
        subtitle = 'subtitle string goes here',
        )

    for s in scans:
        impaths = [os.path.join(inputpath, s, 'images', ch) for ch in channels]
        overviewpath = os.path.join(inputpath, s, 'overview.jpeg')
        scanparam_path = os.path.join(inputpath, s, 'scanparameters.json')
        Story.append(NextPageTemplate('ScanPage'))
        Story = build_scan_page(
            doc = doc,
            Story = Story,
            scan_number = int(s),
            title = 'Scan ' + s,
            text = 'a lot of text to test', 
            scan_params = read_scan_params(scanparam_path),
            overviewmap_image_filepath = overviewpath, 
            scan_image_filepaths = impaths,
            )
    go(doc, Story, outputpath)

inputpath = 'G:\\My Drive\\FRG\\Projects\\APS\\2IDD_2019\\Sample Data - 150C HEP\\export'
outputpath = 'G:\\My Drive\\FRG\\Projects\\APS\\2IDD_2019\\Sample Data - 150C HEP\\report.pdf'
report(outputpath, inputpath)

import subprocess
subprocess.Popen(outputpath ,shell=True)

######

# impath = 'G:\\My Drive\\FRG\\Projects\\APS\\2IDD_2019\\Sample Data - 150C HEP\\export\\67\\67_Al.jpeg'
# impaths = [impath,impath,impath,impath,impath,impath,impath,impath,impath,impath]

# ######

# style = styles["Normal"]

# Story = [Spacer(1,2*inch)]


# Story = build_scan_page_old(Story, 
#     scan_number = 67, 
#     title = 'Test', 
#     text = 'a lot of text to test', 
#     overviewmap_image_filepath = impaths[0], 
#     scan_image_filepaths = impaths[1:4])

# Story = build_scan_page_old(Story, 
#     scan_number = 67, 
#     title = 'Test', 
#     text = 'a lot of text to test', 
#     overviewmap_image_filepath = impaths[0], 
#     scan_image_filepaths = impaths[1:3])

# Story = build_scan_page_old(Story, 
#     scan_number = 67, 
#     title = 'Test', 
#     text = 'a lot of text to test', 
#     overviewmap_image_filepath = impaths[0], 
#     scan_image_filepaths = impaths)

# go(Story)

#############################

# import time
# from reportlab.lib.enums import TA_JUSTIFY
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch


# doc = SimpleDocTemplate("test.pdf",
#                         pagesize=letter,
#                         rightMargin=72, leftMargin=72,
#                         topMargin=72, bottomMargin=18)

# Story = []

# magName = "Pythonista"
# issueNum = 12
# subPrice = "99.00"
# limitedDate = "03/05/2010"
# freeGift = "tin foil hat"

# formatted_time = time.ctime()
# full_name = "SME-UnB"
# address_parts = ["Campus Universitario UnB", "Brasilia-DF, 70910-900"]

# impath = 'G:\\My Drive\\FRG\\Projects\\APS\\2IDD_2019\\Sample Data - 150C HEP\\export\\67\\67_Al.jpeg'

# im = Image(impath, 2 * inch, 2 * inch)
# # im2 = Image(logo2, 8 * inch, 5 * inch)
# # im3 = Image(logo3, 8 * inch, 5 * inch)
# # im4 = Image(logo4, 8 * inch, 5 * inch)
# # im5 = Image(logo5, 8 * inch, 5 * inch)

# Story.append(im)
# # Story.append(im2)
# # Story.append(im3)
# # Story.append(im4)
# # Story.append(im5)

# styles = getSampleStyleSheet()
# styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
# ptext = '<font size=12>%s</font>' % formatted_time

# Story.append(Paragraph(ptext, styles["Normal"]))
# Story.append(Spacer(1, 12))

# ptext = '<font size=12>%s</font>' % full_name
# Story.append(Paragraph(ptext, styles["Normal"]))
# for part in address_parts:
#     ptext = '<font size=12>%s</font>' % part.strip()
#     Story.append(Paragraph(ptext, styles["Normal"]))

# Story.append(Spacer(1, 12))
# Story.append(Spacer(1, 12))
# ptext = '<font size=12>{ % trans Report Energy Monitoring % }</font>'
# Story.append(Paragraph(ptext, styles["Normal"]))

# doc.build(Story)