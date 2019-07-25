from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
import numpy as np
import utils

c = canvas.Canvas("test.pdf", 
	bottomup = 0,
	pagesize = letter)

c.drawString(100,100,"testing testing")

testim = [[1,1,1],[0.5,0.5,0.5], [0,0,0]]
impath = 'G:\\My Drive\\FRG\\Projects\\APS\\2IDD_2019\\Sample Data - 150C HEP\\export\\67\\67_Al.jpeg'


def scale_image(fileish, width: int) -> Image:
    """ scales image with given width. fileish may be file or path """
    img = utils.ImageReader(fileish)
    orig_width, height = img.getSize()
    aspect = height / orig_width
    return Image(fileish, width=width, height=width * aspect) 


im = scale_image(impath, 100)
c.drawImage(im)

 doc.build(Story)

c.showPage()
c.save()
