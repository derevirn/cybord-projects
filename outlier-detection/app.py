from tkinter import Tk, Label, Button, filedialog, StringVar, Entry, messagebox
from tkinter import ttk
from outlier_ae import detector_fit, write_output
from os.path import isdir
import tensorflow as tf


def browse_button(path):
    #global folder_path
    filename = filedialog.askdirectory()
    path.set(filename)
    
def run_button():
    #checking if input and output paths have been set
    if (isdir(input_path.get()) == False) or (isdir(output_path.get()) == False):
        messagebox.showerror(title='Error', message="You haven't selected any valid folders")
        
    else:        
        messagebox.showinfo(title='Info',
                            message='Please wait while the detector is being trained')
       
        preds = detector_fit(input_path.get())
        write_output(preds, input_path.get(), output_path.get())
        messagebox.showinfo(title='Info',
                            message='Operation completed successfully!')
        status_str.set('The outliers and CSV file have been saved in the output folder!')
    


window = Tk()
window.title("Anomaly Detection")
window.geometry('640x320')

title = Label(window, text="Outlier Detection App", font=('Arial', 22))
title.place(x=10, y=10)

#Input folder widgets
input_label = Label(window, text="Input Folder:", font=('Arial',10))
input_label.place(x=10, y=80)

input_path = StringVar(window)
input_field = Entry(window, textvariable=input_path, font=('Arial', 14),
                     width=30)
input_field.place(x=100, y=80)

input_btn = Button(text="Open", command=lambda: browse_button(input_path),
                    font=('Arial'),
                    height=1, width=10)
input_btn.place(x=480, y=77)

#Output folder widgets
output_label = Label(window, text="Output Folder:", font=('Arial',10))
output_label.place(x=10, y=160)

output_path = StringVar(window)
output_field = Entry(window, textvariable=output_path, font=('Arial', 14),
                     width=30)
output_field.place(x=100, y=160)

output_btn = Button(text="Open", command=lambda: browse_button(output_path),
                    font=('Arial'),
                    height=1, width=10)
output_btn.place(x=480, y=157)


#Status label
status_str = StringVar(window)
status_str.set("Select the input and output folder to detect the outlier images.")
status_label = Label(window, textvariable=status_str, font=('Arial Bold', 11))
status_label.place(x=10, y=245)


#Run button
run_btn = Button(window, text="Run", command=run_button,
                   font=('Arial', 12),
                   height=1, width=10)
run_btn.place(x=480, y=240)

if(tf.test.is_gpu_available() == False):
    messagebox.showwarning(title='Warning', message = 'No GPU Available')
    
window.mainloop()




