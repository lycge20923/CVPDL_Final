'''
There are two kinds of parameters needed to be changed:
   (1) The sensitive information for connecting the server: Create .env file and type private information
   (2) The size of the gui : Since the width and height depends on mac pro 14, they need to be changed before using
'''

from pathlib import Path
from tkinter import *
from tkinter import ttk
from tkinter.colorchooser import askcolor
from PIL import ImageGrab

import paramiko
import os
from dotenv import load_dotenv
import time

load_dotenv()


####### (1) Need to be modified #######
HOST_NAME = os.getenv('HOST_NAME')
USERNAME = os.getenv('NAME')
PASSWORD = os.getenv('PASSWORD')
PORT = 22
server_dir = 'Graduate/112-1_CVPDL/CVPDL_Final'
python_path = '/home/r12922166/anaconda3/envs/CVPDL_FINAL/bin/python'
#####################################


# Conndect server through ssh
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=HOST_NAME, port=PORT, username=USERNAME, password=PASSWORD)

# For GUI Paint Block
class Paint(object):
    DEFAULT_COLOR = 'black'
    def __init__(self, window):
        self.root = window
        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)
        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)
        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)
        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)
        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)
        self.c = Canvas(self.root, bg='white', width=1024.0, height=576.0) # (2) Change Size 
        self.c.grid(row=1, columnspan=5)
        self.setup()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

# For GUI 
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def RunFunc():
    style = value.get()
    prompt = entry_2.get()
    
    #the window size needs to be changed to adapt to your computer
    img = ImageGrab.grab(bbox=(entry_1.winfo_x()+10, entry_1.winfo_y()+110, int(entry_1.winfo_x()+1000.0), int(entry_1.winfo_y()+600.0))) # (2) Change Size 

    img_path = "input.jpg"
    server_img_path = os.path.join(server_dir, 'input.jpg')
    img.convert('RGB').save(img_path)
    print(style, prompt)
    if os.path.isfile('video_with_audio.mp4'):
        os.remove('video_with_audio.mp4')
    transport = paramiko.Transport((HOST_NAME,PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(img_path, server_img_path)
    try:
        ssh.exec_command("cd " + server_dir + "; rm video_with_audio.mp4")
    except:
        pass
    
    server_vidoe_path = os.path.join(server_dir, 'video_with_audio.mp4')
    vieo_path = 'video_with_audio.mp4'
    cmd = "cd " + server_dir+"; " + python_path + " main.py"+' --doodle_path input.jpg --style '+"'"+style+"'"+' --prompt '+"'"+prompt+"'"
    print(cmd)
    stdin, stdout, stderr = ssh.exec_command(cmd)
    t = 660
    for i in range(t+1):
        print(f'\r[{"█"*i}{" "*(t-i)}] {round(i*100/t,2)}%', end='')  
        time.sleep(1)
    sftp.get(server_vidoe_path, vieo_path)
    os.system('open '+vieo_path)
    window.destroy()
    print(cmd)

window = Tk()
window.geometry("1512x982")
window.configure(bg = "#FFFFFF")

canvas = Canvas(window, bg = "#FFFFFF", height = 1009, width = 1512, bd = 0, highlightthickness = 0, relief = "ridge") # (2) Change Size 

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image( 756.0, 504.0, image=image_image_1)

canvas.create_text(430.0, 38.0, anchor="nw", text="D.D.V.M:Doodle to Dubbed Video Models", fill="#000000", font=("JollyLodger", 60 * -1)) # (2) Change Size 

canvas.create_text(63.0, 741.0, anchor="nw", text="Prompt", fill="#000000", font=("JollyLodger", 36 * -1)) # (2) Change Size 

canvas.create_text(800.0, 740.0, anchor="nw", text="Style", fill="#000000", font=("JollyLodger", 36 * -1)) # (2) Change Size 

entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(756.1653747558594, 408.0, image=entry_image_1)

entry_1 = Frame(window)
entry_1.pack()
entry_1.place(x=254.0, y=128.0, width=1004.0,height=575.0) # (2) Change Size 
entry_11 = Paint(entry_1)

entry_image_2 = PhotoImage(file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(466.0, 770.0, image=entry_image_2) 
entry_2 = Entry(bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0, font="Helvetica 24 bold")
entry_2.place(x=215.0, y=744.0, width=502.0, height=50.0) # (2) Change Size 

button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
button_1 = Button( image = button_image_1, borderwidth=0, highlightthickness=0, command=RunFunc, relief="flat")
button_1.place( x=495.0, y=824.0, width=522.0, height=79.0) # (2) Change Size 

entry_image_4 = PhotoImage(file=relative_to_assets("entry_4.png"))
entry_bg_4 = canvas.create_image(1188.0, 770.0, image=entry_image_4)
optionList = ['(No style)', 'Cinematic', '3D Model', 'Anime', 'Digital Art', 'Photographic', 'Pixel art', 'Fantasy art', 'Neonpunk', 'Manga']   # 選項
value = StringVar()                                        
menu = ttk.Combobox(window, textvariable=value, values=optionList, height=50, state="readonly", font="Verdana 25 bold")
menu.pack()
menu.place(x=937.0, y=744.0, width=502.0, height=40.0) # (2) Change Size 

window.mainloop()
