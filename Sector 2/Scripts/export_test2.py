from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch
from PIL import Image as pilImage
import numpy as np


PAGE_HEIGHT=defaultPageSize[1]; PAGE_WIDTH=defaultPageSize[0]
styles = getSampleStyleSheet()

Title = "Hello world"
pageinfo = "platypus example"
titlefont = 'Times-Bold'
titlefontsize = 16
bodyfont = 'Times-Roman'
bodyfontsize = 9

def myFirstPage(canvas, doc):
    canvas.saveState()
    canvas.setFont(titlefont,titlefontsize)
    canvas.drawCentredString(PAGE_WIDTH/2.0, PAGE_HEIGHT-108, Title)
    canvas.setFont(bodyfont,bodyfontsize)
    canvas.drawString(inch, 0.75 * inch, "First Page / %s" % pageinfo)
    canvas.restoreState()

def myLaterPages(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman',bodyfontsize)
    canvas.drawString(inch, 0.75 * inch, "Page %d %s" % (doc.page, pageinfo))
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
        height = printheight)
    return image

def generate_image_matrix(image_paths, max_num_cols):
    num_cols = np.min([max_num_cols, len(image_paths)])
    image_width = PAGE_WIDTH / (num_cols+1)

    filling = True
    img_matrix = []
    image_index = 0
    while filling:
        row_matrix = []
        for col in range(num_cols):
            if image_index < len(image_paths):
                im = ScaledImage(image_paths[image_index], dimension = 'width', size = image_width)
                row_matrix.append(im)
                image_index += 1
            else:
                filling = False
                row_matrix.append('')
        img_matrix.append(row_matrix)
    return img_matrix

def build_scan_page(Story, scan_number, title, text, map_image_filepath, scan_image_filepaths):
    headerstr = 'Scan ' + str(scan_number) + ': ' + title
    Story.append(Paragraph(headerstr, styles['Normal']))

    Story.append(Paragraph(text, styles['Normal']))

    imtable = Table(generate_image_matrix(impaths, 4))
    Story.append(imtable)
    Story.append(PageBreak())

    return Story

def go(Story):
    doc = SimpleDocTemplate("phello.pdf")
    doc.build(Story, onFirstPage=myFirstPage, onLaterPages=myLaterPages)

######


Story = [Spacer(1,2*inch)]
style = styles["Normal"]
Story.append(PageBreak())



impath = 'G:\\My Drive\\FRG\\Projects\\APS\\2IDD_2019\\Sample Data - 150C HEP\\export\\67\\67_Al.jpeg'
impaths = [impath,impath,impath,impath,impath,impath,impath,impath,impath,impath]

build_scan_page(Story, 
    scan_number = 67, 
    title = 'Test', 
    text = 'a lot of text to test', 
    map_image_filepath = impaths, 
    scan_image_filepaths = impaths)


build_scan_page(Story, 
    scan_number = 67, 
    title = 'Test', 
    text = 'a lot of text to test', 
    map_image_filepath = impaths, 
    scan_image_filepaths = impaths)


build_scan_page(Story, 
    scan_number = 67, 
    title = 'Test', 
    text = 'a lot of text to test', 
    map_image_filepath = impaths, 
    scan_image_filepaths = impaths)

for i in range(100):
    bogustext = ("This is Paragraph number %s. " % i) *20
p = Paragraph(bogustext, style)
Story.append(p)
Story.append(Spacer(1,0.2*inch))

go(Story)



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