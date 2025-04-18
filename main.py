import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
plt.rcParams['font.family'] = 'Microsoft JhengHei'
from tkinter import *
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Timer
import threading
import gc
import psutil, os
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
except:
    pass

def get_base_dir():
    # PyInstaller 執行時會有 _MEIPASS，否則是原始程式路徑
    if hasattr(sys, '_MEIPASS'):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)

def periodic_memory_cleanup():
    gc.collect()  # 觸發 Python 垃圾回收
    root.after(10000, periodic_memory_cleanup)  # 每 10 秒執行一次

def find_camera_index(max_index=5):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(i)
        cap.release()
    return available
    
def save_image_async(path, frame):
    threading.Thread(target=cv2.imwrite, args=(path, frame)).start()

# ========= 工具函數：打包後找資源的路徑 ==========
def resource_path(filename):
    """
    在開發階段從原本資料夾讀取，在打包後從 PyInstaller 的資料夾中讀取
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(__file__), filename)

# ========= 全域變數 ==========
cap = None
cap2 = None
edge_threshold = 50
min_area = 200
sobel_ksize = 3
recording = False
record_log = []
display_log = []
MAX_DISPLAY = 30
frame_to_save = None
output_dir = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(output_dir, exist_ok=True)
screenshot_dir = None  # 截圖儲存資料夾（與 log_file_path 同位置）

roi_x, roi_y, roi_w, roi_h = 80, 60, 160, 120
drag_start = None
log_file_path = None  # 記錄檔案路徑

# ========= 字體與畫中文 ==========
def get_chinese_font(size=20):
    try:
        return ImageFont.truetype("mingliu.ttc", size)
    except:
        return ImageFont.truetype("/usr/share/fonts/truetype/arphic/ukai.ttc", size)

# ========= UI 初始化 ==========
root = Tk()
root.withdraw()

# ========= 顯示開場 LOGO ==========
def show_logo_then_start_main_ui():
    splash = Toplevel()
    splash.overrideredirect(True)
    splash.geometry("400x400+500+200")

    try:
        logo_path = resource_path("logo.png")
        img = Image.open(logo_path).resize((400, 400))
        logo_img = ImageTk.PhotoImage(img)
        label = Label(splash, image=logo_img)
        label.image = logo_img
        label.pack()
    except Exception as e:
        Label(splash, text="⚠ 無法載入 LOGO", font=("Arial", 20)).pack(expand=True)
        print("載入 LOGO 錯誤：", e)

    def launch_main():
        global cap, cap2
        splash.destroy()
        root.deiconify()
        cap = cv2.VideoCapture(int(entry_main_cam.get()))
        cap2 = cv2.VideoCapture(int(entry_sub_cam.get()))
        update_frame()

    splash.after(5000, launch_main)

show_logo_then_start_main_ui()

root.title("RTR-TP 皺褶檢測系統")
main_frame = Frame(root)
main_frame.pack()

from tkinter import Canvas, Scrollbar

# --- 建立可滾動的 left_panel ---
canvas_frame = Frame(main_frame)
canvas_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

canvas = Canvas(canvas_frame, width=400, height=720)
scrollbar = Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

left_panel = scrollable_frame  # ✅ 替代原本的 left_panel
right_panel = Frame(main_frame)
right_panel.grid(row=0, column=1)

status_var = StringVar()
status_var.set("準備就緒")
status_label = Label(right_panel, textvariable=status_var, fg="green", anchor="w", justify="left", wraplength=380)
status_label.pack(fill='x', padx=5, pady=(0, 5), anchor="n")
video_label = Label(right_panel)
video_label.pack()
video_label2 = Label(right_panel)
video_label2.pack()


# --- 外層容器 ---
info_container = Frame(left_panel)
info_container.pack(fill='x', pady=5)

# 固定 info_frame 寬度為 380（你可自行調整）
info_frame = LabelFrame(info_container, padx=5, pady=5, width=380)
info_frame.pack(fill='x')
info_frame.pack_propagate(False)  # ❗ 阻止內部元件影響寬度

# --- 左邊：資訊顯示區 ---
info_frame = LabelFrame(info_container, padx=5, pady=5)
info_frame.pack(side='left', fill='both', expand=True)

info_header = Frame(info_frame)
info_header.pack(fill='x', pady=(0, 5))

Label(info_header, text="📋 資訊顯示區", fg="black", font=("Arial", 10, "bold")).pack(side='left')


label_edge = Label(info_frame, text=f"邊緣強度：{edge_threshold}", fg="green", font=("Microsoft JhengHei", 12))
label_edge.pack(anchor='w')
label_defects = Label(info_frame, text="偵測區塊數：0", fg="blue", font=("Microsoft JhengHei", 12))
label_defects.pack(anchor='w')
label_wrinkle = Label(info_frame, text="皺褶程度：0.00%", fg="orange", font=("Microsoft JhengHei", 12))
label_wrinkle.pack(anchor='w')

# --- 右邊：太空人圖片 ---
astro_frame = Frame(info_container)
astro_frame.pack(side='left', padx=5)

try:
    astro_path = os.path.join(os.path.dirname(__file__), "123.png")  # 你的太空人圖
    astro_img = Image.open(astro_path).resize((150, 150))  # 大小可調整
    astro_photo = ImageTk.PhotoImage(astro_img)
    astro_label = Label(astro_frame, image=astro_photo)
    astro_label.image = astro_photo  # 避免圖像被垃圾回收
    astro_label.pack()
except Exception as e:
    Label(astro_frame, text="🚀圖失敗").pack()
    print("太空人圖載入失敗:", e)

# 左邊標題與右邊說明按鈕



def show_help():# 參數說明OPL
    help_win = Toplevel(root)
    help_win.title("📝 參數說明")
    help_win.geometry("400x300")
    
    txt = Text(help_win, wrap=WORD)
    txt.pack(expand=True, fill=BOTH, padx=10, pady=10)

    content = """🔧 RTR軟板皺褶軟體使用說明：
本軟體目的 : 本軟體為宜蘭生產處開發，用於監測RTR(特別是TP)的皺褶程度。
軟體功能 : 
1.連續監控皺褶程度、數量。
2.即時圖表顯示
3.LOG檔匯出
4.各項參數可自行調整
5.可調整要識別的區域大小
6.可將調整好的參數存下
7.可設定條件進行截圖(拍夾具編號)
8.請詳閱使用說明~感謝

 參數說明 :    
【邊緣強度】
控制影像邊緣的二值化門檻，數值越高，檢測越嚴格。也就是會偵測到越多皺褶，可以自行調整容許值。

【最小區塊面積】
過濾太小的雜訊，設定一個區塊大小下限。也就是顯示框的密集程度。

【Sobel 核心大小】
如果板面影像很清晰、無雜訊 → 用 ksize = 3 很好
如果板面影像有些雜點 → 試試 ksize = 5 或 7
如果想要微調皺褶精度 → 5~9 是合理範圍，TP用3應該可行。

【ROI 寬度/高度】
調整分析區塊的範圍，設定分析目標的寬高。

可拖曳 ROI 框來移動分析位置。


【框框數量截圖基準】
當出現多少框框，就準備進行夾具(第二攝影機)截圖存檔
舉例 : 設定6，就代表當檢測框出現6個框框或以上，就會準備截圖


【達基準多少秒截圖】
當框框數量滿足截圖基準後，這邊設定要連續偵測到幾秒，才會真正截圖
舉例 : 
【框框數量截圖基準】:6
【達基準多少秒截圖】:2，那麼當檢測區出現6個框框，且維持2秒，就會截圖


【截圖最短間隔（秒）】
當截圖一次後，經過多少秒，再次截圖
注意 : 截圖的參數會影響系統效能，建議使用SSD，或是pcie。



"""

    txt.insert(END, content)
    txt.configure(state='disabled')  # 設為唯讀







# 右邊按鈕


param_frame = LabelFrame(left_panel, text="🛠️ 參數調整區", padx=5, pady=5)
param_frame.pack(pady=5, fill='x')
Label(param_frame, text="邊緣強度").pack(anchor='w')
scale_edge = Scale(param_frame, from_=0, to=255, orient=HORIZONTAL)
scale_edge.set(edge_threshold)
scale_edge.pack(fill='x')
Label(param_frame, text="最小區塊面積").pack(anchor='w')
entry_area = Entry(param_frame, width=10)
entry_area.insert(0, str(min_area))
entry_area.pack(fill='x', pady=2)
Label(param_frame, text="Sobel 核心大小（奇數）").pack(anchor='w')
scale_ksize = Scale(param_frame, from_=1, to=31, resolution=2, orient=HORIZONTAL)
scale_ksize.set(sobel_ksize)
scale_ksize.pack(fill='x')


Label(param_frame, text="框框數量截圖基準").pack(anchor='w')
entry_trigger_count = Entry(param_frame, width=5)
entry_trigger_count.insert(0, "20")
entry_trigger_count.pack(fill='x')

Label(param_frame, text="達基準多少秒截圖").pack(anchor='w')
entry_trigger_time = Entry(param_frame, width=5)
entry_trigger_time.insert(0, "5")
entry_trigger_time.pack(fill='x')

Label(param_frame, text="截圖最短間隔（秒）").pack(anchor='w')
entry_capture_gap = Entry(param_frame, width=5)
entry_capture_gap.insert(0, "10")  # 預設 10 秒
entry_capture_gap.pack(fill='x')

def update_roi():
    global roi_w, roi_h, roi_x, roi_y
    try:
        roi_w = max(10, min(int(entry_w.get()), 320))
        roi_h = max(10, min(int(entry_h.get()), 240))
        roi_x = min(roi_x, 320 - roi_w)
        roi_y = min(roi_y, 240 - roi_h)
    except:
        pass

Label(param_frame, text="⚙ 識別框 寬度 / 高度").pack(anchor='w')

roi_frame = Frame(param_frame)
roi_frame.pack(fill='x', pady=(0, 5))  # 加點間距

entry_w = Entry(roi_frame, width=5)
entry_h = Entry(roi_frame, width=5)
entry_w.insert(0, "160")
entry_h.insert(0, "120")
entry_w.pack(side='left', padx=(0, 5))
entry_h.pack(side='left', padx=(0, 5))

Button(roi_frame, text="更新 識別框 大小", command=update_roi).pack(side='left', padx=(5, 0))






# ===== 功能按鈕區 =====
action_frame = LabelFrame(left_panel, text="🚀 功能區", padx=5, pady=5)
action_frame.pack(pady=5, fill='x')

def save_image():
    global frame_to_save
    if frame_to_save is not None:
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All Files", "*.*")],
            initialfile=f"defect_{time.strftime('%Y%m%d_%H%M%S')}.png",
            title="儲存擷取畫面"
        )
        if filename:
            cv2.imwrite(filename, frame_to_save)
            status_var.set(f"✅ 已儲存：{filename}")
        else:
            status_var.set("❌ 取消儲存")

Button(action_frame, text="📸 儲存畫面", command=save_image).pack(fill='x', pady=3)

#status_var = StringVar()
#status_var.set("準備就緒")


# ===== 攝影機選擇區 =====
camera_frame = LabelFrame(left_panel, text="🎥 攝影機選擇", padx=5, pady=5)
camera_frame.pack(pady=5, fill='x')

Label(camera_frame, text="主攝影機 index").pack(anchor='w')
entry_main_cam = Entry(camera_frame, width=5)
entry_main_cam.insert(0, "1")  # 預設值
entry_main_cam.pack(fill='x')

Label(camera_frame, text="副攝影機 index").pack(anchor='w')
entry_sub_cam = Entry(camera_frame, width=5)
entry_sub_cam.insert(0, "2")  # 預設值
entry_sub_cam.pack(fill='x')

def update_cameras():
    global cap, cap2
    try:
        new_main = int(entry_main_cam.get())
        new_sub = int(entry_sub_cam.get())
        cap.release()
        cap2.release()
        cap = cv2.VideoCapture(new_main)
        cap2 = cv2.VideoCapture(new_sub)
        status_var.set(f"🎥 攝影機已切換為 {new_main} 與 {new_sub}")
    except Exception as e:
        status_var.set(f"⚠️ 攝影機切換失敗: {e}")

Button(camera_frame, text="更新攝影機", command=update_cameras).pack(fill='x', pady=3)

def save_config():
    filename = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
        initialfile="config.txt",
        title="儲存參數設定為..."
    )
    if filename:
        config = {
            "edge_threshold": scale_edge.get(),
            "min_area": entry_area.get(),
            "sobel_ksize": scale_ksize.get(),
            "roi_width": entry_w.get(),
            "roi_height": entry_h.get()
        }
        try:
            with open(filename, "w", encoding="utf-8") as f:
                for k, v in config.items():
                    f.write(f"{k}={v}\n")
            status_var.set(f"✅ 參數已儲存：{filename}")
        except Exception as e:
            status_var.set("⚠️ 儲存失敗")
            print("儲存失敗：", e)
    else:
        status_var.set("❌ 取消儲存")

def load_config():
    global roi_w, roi_h, roi_x, roi_y
    filename = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt")],
        title="選擇要載入的參數設定"
    )
    if filename:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
            config = dict(line.strip().split("=") for line in lines)
            scale_edge.set(int(config["edge_threshold"]))
            entry_area.delete(0, END)
            entry_area.insert(0, config["min_area"])
            scale_ksize.set(int(config["sobel_ksize"]))
            entry_w.delete(0, END)
            entry_h.delete(0, END)
            entry_w.insert(0, config["roi_width"])
            entry_h.insert(0, config["roi_height"])
            update_roi()
            status_var.set(f"✅ 已載入參數：{os.path.basename(filename)}")
        except Exception as e:
            status_var.set("⚠️ 載入失敗")
            print("載入失敗：", e)
    else:
        status_var.set("❌ 取消載入")

Button(action_frame, text="💾 儲存參數", command=save_config).pack(fill='x', pady=2)
Button(action_frame, text="📂 載入參數", command=load_config).pack(fill='x', pady=2)

def toggle_record():
    global recording, record_log, display_log, log_file_path, screenshot_dir
    if not recording:
        log_file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile=f"wrinkle_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            title="儲存 LOG 至..."
        )
        if not log_file_path:
            status_var.set("❌ 取消開始記錄")
            return

        # 依照 log 檔案路徑，自動建立截圖資料夾
        screenshot_dir = os.path.join(get_base_dir(), "captures")
        os.makedirs(screenshot_dir, exist_ok=True)

        # 初始化 LOG 檔案
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("區塊數,皺褶%,時間\n")

        record_log.clear()
        display_log.clear()
        recording = True
        record_btn.config(text="■ 停止記錄")
        show_chart()
        status_var.set(f"📈 開始記錄中... 檔案：{os.path.basename(log_file_path)}")
    else:
        recording = False
        status_var.set("🛑 停止記錄")
        record_btn.config(text="▶ 開始記錄")

# 替換原本的按鈕
record_btn = Button(action_frame, text="▶ 開始記錄", command=toggle_record)
record_btn.pack(fill='x')

#Label(action_frame, textvariable=status_var, fg="green").pack(pady=5),不用喔，會影響UI



# ========= 圖表 ==========
def show_chart():
    chart_win = Toplevel(root)
    chart_win.title("📈 即時圖表")
    fig = plt.Figure(figsize=(5, 2), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, chart_win)
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def update_chart():
        if len(display_log) > 1:
            ax.clear()
            x_vals = list(range(1, len(display_log) + 1))  # 用 1, 2, 3... 表示第幾筆
            ax.plot(x_vals, [v[0] for v in display_log], label="區塊數", color='blue')
            ax.plot(x_vals, [v[1] for v in display_log], label="皺褶%", color='orange')
            ax.set_xlabel("資料筆數")  # 如果你想加一個 X 軸標題
            ax.legend()
            canvas.draw()
        chart_win.after(2000, update_chart)

    update_chart()

# ========= ROI 拖曳 ==========
def start_drag(event):
    global drag_start
    if roi_x <= event.x <= roi_x + roi_w and roi_y <= event.y <= roi_y + roi_h:
        drag_start = (event.x, event.y)

def drag_roi(event):
    global roi_x, roi_y, drag_start
    if drag_start:
        dx = event.x - drag_start[0]
        dy = event.y - drag_start[1]
        roi_x = max(0, min(roi_x + dx, 320 - roi_w))
        roi_y = max(0, min(roi_y + dy, 240 - roi_h))
        drag_start = (event.x, event.y)

def end_drag(event):
    global drag_start
    drag_start = None

video_label.bind("<Button-1>", start_drag)
video_label.bind("<B1-Motion>", drag_roi)
video_label.bind("<ButtonRelease-1>", end_drag)

# ========= 攝影機處理 ==========暫時不用了，我讓人員自己選擇
#cap = cv2.VideoCapture(1)
#cap2 = cv2.VideoCapture(2)

def update_frame():
    global frame_to_save


    
    ret, frame = cap.read()
    if not ret:
        root.after(200, update_frame)
        return
    

    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    ksize = scale_ksize.get()
    sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    _, sobel_thresh = cv2.threshold(sobel, scale_edge.get(), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(sobel_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_count = 0
    annotated = frame.copy()

    try:
        min_area_val = max(0, int(entry_area.get()))
    except:
        min_area_val = 0  # 若輸入錯誤，預設為 0

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area_val:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (roi_x + x, roi_y + y), (roi_x + x + w, roi_y + y + h), (0, 0, 255), 2)
            defect_count += 1

    white = cv2.countNonZero(sobel_thresh)
    wrinkle = (white / (sobel_thresh.shape[0] * sobel_thresh.shape[1])) * 100

    cv2.rectangle(annotated, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
    frame_to_save = annotated.copy()

    label_edge.config(text=f"邊緣強度：{scale_edge.get()}")
    label_defects.config(text=f"偵測區塊數：{defect_count}")
    label_wrinkle.config(text=f"皺褶程度：{wrinkle:.2f}%")
    
    if recording:
        ts = time.time()
        if not hasattr(update_frame, "last_record_time") or int(ts) != int(update_frame.last_record_time):
            record_log.append((defect_count, wrinkle, ts))
            display_log.append((defect_count, wrinkle, ts))
            if log_file_path:
                 with open(log_file_path, "a", encoding="utf-8") as f:
                   f.write(f"{defect_count},{wrinkle:.2f},{time.strftime('%H:%M:%S', time.localtime(ts))}\n")
            update_frame.last_record_time = ts

    if len(display_log) > MAX_DISPLAY:
        display_log[:] = display_log[-MAX_DISPLAY:]

    sobel_bgr = cv2.cvtColor(cv2.resize(sobel, (320, 240)), cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(cv2.resize(sobel_thresh, (320, 240)), cv2.COLOR_GRAY2BGR)
    gray_bgr = cv2.cvtColor(cv2.resize(gray, (320, 240)), cv2.COLOR_GRAY2BGR)
    top = np.hstack((annotated, sobel_bgr))
    bottom = np.hstack((gray_bgr, thresh_bgr))
    combined = np.vstack((top, bottom))

    img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)


# ========== 第二攝影機邏輯 ==========
    ret2, frame2 = cap2.read()
    if ret2:
        frame2 = cv2.resize(frame2, (320, 240))
        img2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        imgtk2 = ImageTk.PhotoImage(image=img2)
        video_label2.imgtk = imgtk2
        video_label2.configure(image=imgtk2)

    # ====== 判斷是否達成異常條件 ======
    try:
        trigger_count = int(entry_trigger_count.get())
        trigger_time = int(entry_trigger_time.get())
    except:
        trigger_count, trigger_time = 9999, 9999

    try:
        capture_gap = int(entry_capture_gap.get())
    except:
        capture_gap = 10  # 預設 10 秒

    now = time.time()

    if recording and defect_count >= trigger_count:
        if not hasattr(update_frame, "start_time") or update_frame.start_time is None:
            update_frame.start_time = now
            update_frame.last_capture_time = 0  # 初始化
        elif now - update_frame.start_time >= trigger_time:
            if not hasattr(update_frame, "last_capture_time"):
                update_frame.last_capture_time = 0
            if now - update_frame.last_capture_time >= capture_gap:
                if screenshot_dir:
                    filename = f"jig_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    path = os.path.join(screenshot_dir, filename)
                    cv2.imwrite(path, frame2)
                    status_var.set(f"📸 已截圖夾具畫面：{filename}")
                    update_frame.last_capture_time = now
    else:
        update_frame.start_time = None
    root.after(50, update_frame)

root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))

periodic_memory_cleanup()
root.mainloop()