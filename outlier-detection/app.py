from tkinter import Tk, Label, Button, filedialog, StringVar
from outlier_ae import detector_fit
from collections import Counter

def browse_button():
    #global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    
def train_button():
    preds = detector_fit(folder_path.get())
    counter = Counter(preds['data']['is_outlier'])
    outliers = dict(counter)
    outliers_str.set("Number of outliers: " + str(outliers[1]))
    

window = Tk()
window.title("Anomaly Detection")
window.geometry('800x600')


title = Label(window, text="Anomaly Detection App", font=('Arial', 24))
title.place(x=10, y=10)

folder_path = StringVar(window)
folder_lbl = Label(window, textvariable=folder_path, font=('Arial', 16))
folder_lbl.place(x=10, y=80)

outliers_str = StringVar(window)
outliers_lbl = Label(window, textvariable=outliers_str, font=('Arial', 16))
outliers_lbl.place(x=10, y=120)

browse_btn = Button(text="Select Folder", command=browse_button,
                    font=('Arial', 12),
                    height=4, width=20)
browse_btn.place(x=500, y=50)


train_btn = Button(window, text="Click to train the detector", command=train_button,
                   font=('Arial', 12),
                   height=4, width=20)
train_btn.place(x=500, y=450)

window.mainloop()


