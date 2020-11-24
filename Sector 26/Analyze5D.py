#!/usr/local/bin python3
"""addon for analyze5d"""

import os
from gi.repository import Gtk
import numpy as np
from matplotlib import pyplot, colors
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from misc_dude import * 
from multiprocessing import Pool, cpu_count

def Display2Data(Axe, x, y):
    """display coordinate to data coordinate"""

    return Axe.transData.inverted().transform(np.array([(x,y)]))[0]


class MainWindow:

    def Generate(self, widget, dude):

        ymax = int(dude.Scan_ToolBox_CustomROI_YMax_Spin_Adjustment.get_value())
        ymin = int(dude.Scan_ToolBox_CustomROI_YMin_Spin_Adjustment.get_value())
        xmax = int(dude.Scan_ToolBox_CustomROI_XMax_Spin_Adjustment.get_value())
        xmin = int(dude.Scan_ToolBox_CustomROI_XMin_Spin_Adjustment.get_value())   
        shifts = np.loadtxt(os.path.join(self.analysis_folder, "shifts.txt"), dtype=int)
        gamma = np.radians(np.genfromtxt(os.path.join(self.analysis_folder, "gamma_mot.csv"), delimiter=',')[ymin:ymax+1, xmin:xmax+1])
        twotheta = np.radians(np.genfromtxt(os.path.join(self.analysis_folder, "twotheta_mot.csv"), delimiter=',')[ymin:ymax+1, xmin:xmax+1])
        theta = (np.pi/2-np.radians(shifts[:,1])/1000.)[:,np.newaxis,np.newaxis]
        self.data5d = np.zeros((shifts.shape[0], shifts[0,5]-shifts[0,4]+1, shifts[0,7]-shifts[0,6]+1, ymax-ymin+1, xmax-xmin+1))
        K = 2*3.14/(12.398/float(self.Energy_Entry.get_text()))
        self.qy = K*np.sin(gamma)*np.ones(theta.shape)
        self.qx = K*np.cos(gamma)*np.sin(twotheta)
        self.qz = K*np.cos(gamma)*np.cos(twotheta)-K
        self.Q  = np.sqrt(self.qx**2+self.qy**2+self.qz**2)
        self.qz, self.qx = np.cos(theta)*self.qz-np.sin(theta)*self.qx, np.sin(theta)*self.qz+np.cos(theta)*self.qx
        # rotate by theta-90, so that qx is along sample x, qy is along sample y, qz is negative sample normal
        
        for i in range(shifts.shape[0]):
            print("grabbing data from scan {0}".format(shifts[i,0]))
            image_path = os.path.join(dude.Image_folder, str(shifts[i,0]))
            image_list = [imagefile.name for imagefile in os.scandir(image_path) if imagefile.name.endswith('.tif')]
            tif_index = np.array([int(filename.split(".")[0].split("_")[-1]) for filename in image_list])
            image_list = np.take(image_list , tif_index.argsort())
            image_list = list(map(lambda x:os.path.join(image_path,x), image_list))
            datatmp = np.zeros((len(image_list),self.data5d.shape[3], self.data5d.shape[4]), dtype="float64")
            pool = Pool(processes=cpu_count()) 
            for j, tmparr in enumerate(pool.imap(image_loader, image_list)):
                datatmp[j] = tmparr[ymin:ymax+1, xmin:xmax+1]
            pool.close()
            self.data5d[i] = datatmp.reshape(shifts[i,8], shifts[i,9], self.data5d.shape[3], self.data5d.shape[4])[shifts[i,4]:shifts[i,5]+1, shifts[i,6]:shifts[i,7]+1]
        np.savez_compressed(os.path.join(self.analysis_folder, "data5D.npz"), data = self.data5d, qx=self.qx, qy=self.qy, qz=self.qz)
        self.data3d = np.copy(self.data5d[:,0,0])
        self.Plot_Init()


    def Save(self, widget):
        
        np.savez_compressed(os.path.join(self.analysis_folder, "data5D.npz"), data = self.data5d, qx=self.qx, qy=self.qy, qz=self.qz)
    

    def Load(self, widget, dude):

        d5 = np.load(os.path.join(self.analysis_folder, "data5D.npz"))
        self.data5d = d5["data"]
        self.qx = d5["qx"]
        self.qy = d5["qy"]
        self.qz = d5["qz"]
        self.data3d = np.copy(self.data5d[:,0,0])
        self.Plot_Init()


    def CoM(self, widget):

        self.qx_com = (self.data3d * self.qx).sum()/self.data3d.sum()
        self.qy_com = (self.data3d * self.qy).sum()/self.data3d.sum()
        self.qz_com = (self.data3d * self.qz).sum()/self.data3d.sum()
        self.CoM_Label.set_text("Qx[{0:.3f}] Qy[{1:.3f}] Qz[{2:.3f}]".format(self.qx_com, self.qy_com, self.qz_com))


    def Correct(self, widget):

        # qy = 0
        if np.fabs(self.qx_com) > 1e-2:
            t = np.arctan2(-self.qy_com, self.qx_com)
            #print (self.qx_com*np.cos(t)-self.qy_com*np.sin(t), self.qx_com*np.sin(t)+self.qy_com*np.cos(t))
            self.qx, self.qy = self.qx*np.cos(t)-self.qy*np.sin(t), self.qx*np.sin(t)+self.qy*np.cos(t)
            self.qx_com = self.qx_com*np.cos(t)-self.qy_com*np.sin(t)
        # qx = 0
        t = np.arctan2(-self.qx_com, -self.qz_com)
        print (t/3.14*180)
        #print (self.qz_com*np.cos(t)+self.qx_com*np.sin(t), -self.qz_com*np.sin(t)+self.qx_com*np.cos(t))
        self.qz, self.qx = self.qz*np.cos(t)+self.qx*np.sin(t), -self.qz*np.sin(t)+self.qx*np.cos(t)
        self.qz_com = self.qz_com*np.cos(t)+self.qx_com*np.sin(t)

        
    def Calculate(self, widget):

        qx_com = (self.data5d * self.qx[:,np.newaxis,np.newaxis,:,:]).sum(3).sum(3).sum(0)/self.data5d.sum(3).sum(3).sum(0)
        qy_com = (self.data5d * self.qy[:,np.newaxis,np.newaxis,:,:]).sum(3).sum(3).sum(0)/self.data5d.sum(3).sum(3).sum(0)
        qz_com = (self.data5d * self.qz[:,np.newaxis,np.newaxis,:,:]).sum(3).sum(3).sum(0)/self.data5d.sum(3).sum(3).sum(0)
        q_com = np.sqrt(qx_com**2+qy_com**2+qz_com**2)
        strain = (q_com+self.qz_com)/-self.qz_com
        self.Strain_Min_Entry.set_text("{0:.5f}".format(strain.min()))
        self.Strain_Max_Entry.set_text("{0:.5f}".format(strain.max()))
        try:
            self.Result_Image.set_array(strain)
        except:
            self.Result_Image = self.Result_Axe.imshow(strain, cmap='coolwarm', aspect='equal', origin='lower', interpolation = 'nearest')
        self.Result_Image.set_norm(colors.Normalize())
        try:
            self.Result_Quiver.remove()
            del self.Result_Quiver
        except Exception as e:
            print (e)
        x, y = np.meshgrid(np.arange(self.data5d.shape[2]),np.arange(self.data5d.shape[1]))
        bins = int(self.Tilt_Bin_Entry.get_text())
        ny = int(y.shape[0]/bins)
        nx = int(y.shape[1]/bins)
        filter_i = self.RealSpace_Image.get_array() > float(self.Tilt_Filter_Entry.get_text())
        self.Result_Quiver = self.Result_Axe.quiver(x[:ny*bins,:nx*bins].reshape(ny,bins,nx,bins).mean(3).mean(1), \
                                                    y[:ny*bins,:nx*bins].reshape(ny,bins,nx,bins).mean(3).mean(1), \
                                                    (qx_com*filter_i)[:ny*bins,:nx*bins].reshape(ny,bins,nx,bins).mean(3).mean(1),\
                                                    (qy_com*filter_i)[:ny*bins,:nx*bins].reshape(ny,bins,nx,bins).mean(3).mean(1),\
                                                    scale = float(self.Tilt_Scale_Entry.get_text()),\
                                                    width = float(self.Tilt_Width_Entry.get_text()))
        self.Result_Canvas.draw()
        

    def Strain_Vscale_Changed(self, widget):

        try:
            self.Result_Image.set_norm(colors.Normalize(float(self.Strain_Min_Entry.get_text()), float(self.Strain_Max_Entry.get_text())))
        except:
            pass
        else:
            self.Result_Canvas.draw()


    def Plot_Init(self):

        try:
            self.RealSpace_Image.set_array(self.data5d.sum(4).sum(3).sum(0))
        except:
            self.RealSpace_Image = self.RealSpace_Axe.imshow(self.data5d.sum(4).sum(3).sum(0), cmap='jet', aspect='equal', origin='lower', interpolation = 'nearest')
        self.RealSpace_Image.set_norm(colors.LogNorm())
        self.RealSpace_Canvas.draw()
        vmin = self.data3d.min()
        vmax = self.data3d.max()
        self.Qxy_Axe.cla()
        self.Qxy_Axe.set_axis_off()
        self.Qxy_Image = self.Qxy_Axe.scatter(self.qx.flatten(), self.qy.flatten(), c=self.data3d.flatten(), norm=colors.LogNorm(vmin = vmin+.1, vmax = vmax+.1), edgecolor='none', cmap='jet')
        self.Qxz_Axe.cla()
        self.Qxz_Axe.set_axis_off()
        self.Qxz_Image = self.Qxz_Axe.scatter(self.qx.flatten(), self.qz.flatten(), c=self.data3d.flatten(), norm=colors.LogNorm(vmin = vmin+.1, vmax = vmax+.1), edgecolor='none', cmap='jet')
        self.Qyz_Axe.cla()
        self.Qyz_Axe.set_axis_off()
        self.Qyz_Image = self.Qyz_Axe.scatter(self.qy.flatten(), self.qz.flatten(), c=self.data3d.flatten(), norm=colors.LogNorm(vmin = vmin+.1, vmax = vmax+.1), edgecolor='none', cmap='jet')
        self.Qxy_Canvas.draw()
        self.Qxz_Canvas.draw()
        self.Qyz_Canvas.draw()
        self.Q_Vmin_HScale_Adjustment.set_lower(vmin)
        self.Q_Vmax_HScale_Adjustment.set_upper(vmax)
        self.Q_Vmin_HScale_Adjustment.handler_block(self.Q_Vmin_Changed_Handler)
        self.Q_Vmax_HScale_Adjustment.handler_block(self.Q_Vmax_Changed_Handler)
        self.Q_Vmin_HScale_Adjustment.set_upper(vmax-1)
        self.Q_Vmin_HScale_Adjustment.set_value(vmin)
        self.Q_Vmax_HScale_Adjustment.set_lower(vmin+1)
        self.Q_Vmax_HScale_Adjustment.set_value(vmax)
        self.Q_Vmin_HScale_Adjustment.handler_unblock(self.Q_Vmin_Changed_Handler)
        self.Q_Vmax_HScale_Adjustment.handler_unblock(self.Q_Vmax_Changed_Handler)

        
    def Q_Vscale_Changed(self, widget):

        vmin = self.Q_Vmin_HScale_Adjustment.get_value()
        vmax = self.Q_Vmax_HScale_Adjustment.get_value()
        self.Q_Vmax_HScale_Adjustment.set_lower(vmin+1)
        self.Q_Vmin_HScale_Adjustment.set_upper(vmax-1)
        sizes = (self.data3d.flatten()<=vmax) * (self.data3d.flatten()>=vmin) * 20
        self.Qxy_Image.set_sizes(sizes)
        self.Qxz_Image.set_sizes(sizes)
        self.Qyz_Image.set_sizes(sizes)
        self.Qxy_Image.set_norm(colors.LogNorm(vmin = vmin+.1, vmax = vmax+.1))
        self.Qxz_Image.set_norm(colors.LogNorm(vmin = vmin+.1, vmax = vmax+.1))
        self.Qyz_Image.set_norm(colors.LogNorm(vmin = vmin+.1, vmax = vmax+.1))
        self.Qxy_Canvas.draw()
        self.Qxz_Canvas.draw()
        self.Qyz_Canvas.draw()

        
    def RealSpace_Canvas_Button_Pressed(self, event):

        self.real_xstart, self.real_ystart = list(map(lambda x: int(round(x, 0)), (Display2Data(self.RealSpace_Axe, event.x, event.y))))

        dim_y, dim_x = self.data5d.shape[1], self.data5d.shape[2]

        if (self.real_ystart < dim_y-0.5) and (self.real_ystart > -.5) and (self.real_xstart > -0.5) and (self.real_xstart < dim_x-0.5):
            if event.button == 3:
                self.data3d = np.copy(self.data5d[:,self.real_ystart,self.real_xstart])
                self.Plot_Init()
            elif event.button == 1:
                self.RealSpace_Canvas.mpl_disconnect(self.RealSpace_Canvas_Button_Press_Event)
                self.RealSpace_Canvas_Button_Release_Event = self.RealSpace_Canvas.mpl_connect('button_release_event', self.RealSpace_Canvas_Button_Released)
       

    def RealSpace_Canvas_Button_Released(self, event):

        self.real_xend, self.real_yend = list(map(lambda x: int(round(x, 0)), (Display2Data(self.RealSpace_Axe, event.x, event.y))))

        dim_y, dim_x = self.data5d.shape[1], self.data5d.shape[2]

        self.RealSpace_Canvas.mpl_disconnect(self.RealSpace_Canvas_Button_Release_Event)
        self.RealSpace_Canvas_Button_Press_Event = self.RealSpace_Canvas.mpl_connect('button_press_event', self.RealSpace_Canvas_Button_Pressed)

        if (self.real_ystart < dim_y-0.5) and (self.real_ystart > -.5) and (self.real_xstart > -0.5) and (self.real_xstart < dim_x-0.5):
            self.data3d = self.data5d[:,min(self.real_ystart, self.real_yend): max(self.real_ystart, self.real_yend)+1, min(self.real_xstart, self.real_xend): max(self.real_xstart, self.real_xend)+1].sum(1).sum(1)
            self.Plot_Init()

                
    def Notebook_Switched(self, widget, number):
        
        if widget.get_active():
            self.ReciprocalSpace_Notebook.set_current_page(number)

            
    def __init__(self, dude):

        self.RealSpace_Figure = Figure()
        self.RealSpace_Axe = self.RealSpace_Figure.add_axes([0, 0, 1, 1])
        self.RealSpace_Axe.set_axis_off()
        self.RealSpace_Canvas = FigureCanvas(self.RealSpace_Figure)
        self.RealSpace_ScrolledWindow = Gtk.ScrolledWindow()
        self.RealSpace_ScrolledWindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.RealSpace_ScrolledWindow.add(self.RealSpace_Canvas)
        self.RealSpace_ScrolledWindow_EventBox = Gtk.EventBox()
        self.RealSpace_ScrolledWindow_EventBox.add(self.RealSpace_ScrolledWindow)
        self.RealSpace_ScrolledWindow.set_size_request(400,400)

        self.RealSpace_Canvas_Button_Press_Event = self.RealSpace_Canvas.mpl_connect('button_press_event', self.RealSpace_Canvas_Button_Pressed)

        self.Result_Figure = Figure()
        self.Result_Axe = self.Result_Figure.add_axes([0, 0, 1, 1])
        self.Result_Axe.set_axis_off()
        self.Result_Canvas = FigureCanvas(self.Result_Figure)
        self.Result_ScrolledWindow = Gtk.ScrolledWindow()
        self.Result_ScrolledWindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.Result_ScrolledWindow.add(self.Result_Canvas)
        self.Result_ScrolledWindow_EventBox = Gtk.EventBox()
        self.Result_ScrolledWindow_EventBox.add(self.Result_ScrolledWindow)
        self.Result_ScrolledWindow.set_size_request(400,400)
        
        self.Qxy_Figure = Figure()
        self.Qxy_Axe = self.Qxy_Figure.add_axes([0, 0, 1, 1])
        self.Qxy_Axe.set_axis_off()
        self.Qxy_Canvas = FigureCanvas(self.Qxy_Figure)
        self.Qxz_Figure = Figure()
        self.Qxz_Axe = self.Qxz_Figure.add_axes([0, 0, 1, 1])
        self.Qxz_Axe.set_axis_off()
        self.Qxz_Canvas = FigureCanvas(self.Qxz_Figure)
        self.Qyz_Figure = Figure()
        self.Qyz_Axe = self.Qyz_Figure.add_axes([0, 0, 1, 1])
        self.Qyz_Axe.set_axis_off()
        self.Qyz_Canvas = FigureCanvas(self.Qyz_Figure)

        self.ReciprocalSpace_Notebook = Gtk.Notebook()
        self.ReciprocalSpace_Notebook.set_show_tabs(False)
        self.ReciprocalSpace_Notebook.append_page(self.Qxy_Canvas)
        self.ReciprocalSpace_Notebook.append_page(self.Qxz_Canvas)
        self.ReciprocalSpace_Notebook.append_page(self.Qyz_Canvas)
        self.ReciprocalSpace_Notebook.set_size_request(800,800)
        self.ReciprocalSpace_Notebook.set_current_page(0)

        Energy_Label = Gtk.Label(" Energy (keV) ")
        self.Energy_Entry = Gtk.Entry()
        self.Energy_Entry.set_text(str(10.4))
        self.Energy_Entry.set_width_chars(5)
        
        Generate_Button = Gtk.Button("Generate")
        Generate_Button.set_tooltip_text("Take the shift correction file and angular values of each pixel, to produce the 5D space.")
        Generate_Button.connect("clicked", self.Generate, dude)

        Load_Button = Gtk.Button("Load")
        Load_Button.set_tooltip_text("...")
        Load_Button.connect("clicked", self.Load, dude)

        Save_Button = Gtk.Button("Save")
        Save_Button.set_tooltip_text("...")
        Save_Button.connect("clicked", self.Save)

        CoM_Button = Gtk.Button("CoM")
        CoM_Button.set_tooltip_text("...")
        CoM_Button.connect("clicked", self.CoM)
        self.CoM_Label = Gtk.Label()
        Correct_Button = Gtk.Button("Correct")
        Correct_Button.set_tooltip_text("...")
        Correct_Button.connect("clicked", self.Correct)

        Calculate_Button = Gtk.Button(" Calc ")
        Calculate_Button.set_tooltip_text("...")
        Calculate_Button.connect("clicked", self.Calculate)
        Strain_Label = Gtk.Label(" Strain: ")
        Strain_Min_Label = Gtk.Label(" Min: ")
        self.Strain_Min_Entry = Gtk.Entry()
        self.Strain_Min_Entry.set_width_chars(5)
        Strain_Max_Label = Gtk.Label(" Max: ")
        self.Strain_Max_Entry = Gtk.Entry()
        self.Strain_Max_Entry.set_width_chars(5)
        self.Strain_Min_Entry.connect("changed", self.Strain_Vscale_Changed)
        self.Strain_Max_Entry.connect("changed", self.Strain_Vscale_Changed)

        Tilt_Label = Gtk.Label(" Tilt ")
        Tilt_Bin_Label = Gtk.Label(" Bin: ")
        self.Tilt_Bin_Entry = Gtk.Entry()
        self.Tilt_Bin_Entry.set_width_chars(5)
        self.Tilt_Bin_Entry.set_text('1')
        Tilt_Scale_Label = Gtk.Label(" Scale: ")
        self.Tilt_Scale_Entry = Gtk.Entry()
        self.Tilt_Scale_Entry.set_width_chars(5)
        self.Tilt_Scale_Entry.set_text('.8')
        Tilt_Width_Label = Gtk.Label(" Width: ")
        self.Tilt_Width_Entry = Gtk.Entry()
        self.Tilt_Width_Entry.set_width_chars(7)
        self.Tilt_Width_Entry.set_text('0.004')
        Tilt_Filter_Label = Gtk.Label(" Filter: ")
        self.Tilt_Filter_Entry = Gtk.Entry()
        self.Tilt_Filter_Entry.set_width_chars(7)
        self.Tilt_Filter_Entry.set_text('1e4')

        View_Button = Gtk.Button(label="View")
        self.Qxy_RadioButton = Gtk.RadioButton(group=None, label="Qxy")
        self.Qxy_RadioButton.connect("clicked", self.Notebook_Switched, 0)
        self.Qxz_RadioButton = Gtk.RadioButton(group=self.Qxy_RadioButton, label="Qxz")
        self.Qxz_RadioButton.connect("clicked", self.Notebook_Switched, 1)
        self.Qyz_RadioButton = Gtk.RadioButton(group=self.Qxy_RadioButton, label="Qyz")
        self.Qyz_RadioButton.connect("clicked", self.Notebook_Switched, 2)
        self.View_Label = Gtk.Label()

        self.Q_Vmin_HScale_Adjustment = Gtk.Adjustment(value = 0, lower = 0, upper = 10000, step_increment = 1, page_increment = 10, page_size = 0)
        self.Q_Vmin_Changed_Handler = self.Q_Vmin_HScale_Adjustment.connect("value_changed", self.Q_Vscale_Changed)
        self.Q_Vmin_HScale = Gtk.Scale(orientation = Gtk.Orientation.HORIZONTAL)
        self.Q_Vmin_HScale.set_adjustment(self.Q_Vmin_HScale_Adjustment)
        self.Q_Vmin_HScale.set_size_request(250, 20)
        self.Q_Vmin_HScale.set_value_pos(Gtk.PositionType.LEFT)
        self.Q_Vmin_HScale.set_digits(0)
        Q_Vmin_Label = Gtk.Button('Vmin:')

        self.Q_Vmax_HScale_Adjustment = Gtk.Adjustment(value = 10000, lower = 10000, upper = 20000, step_increment = 1, page_increment = 10, page_size = 0)
        self.Q_Vmax_Changed_Handler = self.Q_Vmax_HScale_Adjustment.connect("value_changed", self.Q_Vscale_Changed)
        self.Q_Vmax_HScale = Gtk.HScale()
        self.Q_Vmax_HScale.set_adjustment(self.Q_Vmax_HScale_Adjustment)
        self.Q_Vmax_HScale.set_size_request(250, 20)
        self.Q_Vmax_HScale.set_value_pos(Gtk.PositionType.LEFT)
        self.Q_Vmax_HScale.set_digits(0)
        Q_Vmax_Label = Gtk.Button('Vmax:')

        HBox1 = Gtk.HBox(homogeneous = False, spacing = 0)
        HBox1.pack_start(Energy_Label, False, False, 0)
        HBox1.pack_start(self.Energy_Entry, True, True, 0)
        HBox1.pack_start(Generate_Button, False, False, 0)
        HBox1.pack_start(Load_Button, False, False, 0)
        HBox1.pack_start(Save_Button, False, False, 0)

        HBox2 = Gtk.HBox(homogeneous = False, spacing = 0)
        HBox2.pack_start(View_Button, False, False, 0)
        HBox2.pack_start(self.Qxy_RadioButton, False, False, 0)
        HBox2.pack_start(self.Qxz_RadioButton, False, False, 0)
        HBox2.pack_start(self.Qyz_RadioButton, False, False, 0)
        HBox2.pack_end(self.View_Label, False, False, 0)

        HBox3 = Gtk.HBox(homogeneous = False, spacing = 0)
        HBox3.pack_start(Q_Vmin_Label, False, False, 0)
        HBox3.pack_start(self.Q_Vmin_HScale, True, True, 0)
        HBox3.pack_start(Q_Vmax_Label, False, False, 0)
        HBox3.pack_start(self.Q_Vmax_HScale, True, True, 0)

        HBox4 = Gtk.HBox(homogeneous = False, spacing = 0)
        HBox4.pack_start(CoM_Button, False, False, 0)
        HBox4.pack_start(self.CoM_Label, True, True, 0)
        HBox4.pack_start(Correct_Button, False, False, 0)

        HBox5 = Gtk.HBox(homogeneous = False, spacing = 0)
        HBox5.pack_start(Calculate_Button, False, False, 0)
        HBox5.pack_start(Strain_Label, False, False, 0)
        HBox5.pack_start(Strain_Min_Label, False, False, 0)
        HBox5.pack_start(self.Strain_Min_Entry, True, True, 0)
        HBox5.pack_start(Strain_Max_Label, False, False, 0)
        HBox5.pack_start(self.Strain_Max_Entry, True, True, 0)

        HBox6 = Gtk.HBox(homogeneous = False, spacing = 0)
        HBox6.pack_start(Tilt_Label, False, False, 0)
        HBox6.pack_start(Tilt_Bin_Label, False, False, 0)
        HBox6.pack_start(self.Tilt_Bin_Entry, False, False, 0)
        HBox6.pack_start(Tilt_Scale_Label, False, False, 0)
        HBox6.pack_start(self.Tilt_Scale_Entry, False, False, 0)
        HBox6.pack_start(Tilt_Width_Label, False, False, 0)
        HBox6.pack_start(self.Tilt_Width_Entry, False, False, 0)
        HBox6.pack_start(Tilt_Filter_Label, False, False, 0)
        HBox6.pack_start(self.Tilt_Filter_Entry, False, False, 0)

        VBox1 = Gtk.VBox(homogeneous = False, spacing = 0)
        VBox1.pack_start(self.RealSpace_ScrolledWindow_EventBox, False, False, 0)
        VBox1.pack_start(self.Result_ScrolledWindow_EventBox, False, False, 0)
        VBox1.pack_start(HBox1, False, False, 0)
        VBox1.pack_start(HBox4, False, False, 0)
        VBox1.pack_start(HBox5, False, False, 0)

        VBox2 = Gtk.VBox(homogeneous = False, spacing = 0)
        VBox2.pack_start(self.ReciprocalSpace_Notebook, False, False, 0)
        VBox2.pack_start(HBox2, False, False, 0)
        VBox2.pack_start(HBox3, False, False, 0)
        VBox2.pack_start(HBox6, False, False, 0)

        HBox0 = Gtk.HBox(homogeneous = False, spacing = 0)
        HBox0.pack_start(VBox1, False, False, 0)
        HBox0.pack_start(VBox2, False, False, 0)

        self.analysis_folder = os.path.join(os.path.abspath(os.path.join(dude.MDA_folder, os.pardir)),"Analysis")

        self.win = Gtk.Window()
        self.win.add(HBox0)
        self.win.set_title("Analyze 5D")
        self.win.show_all()
