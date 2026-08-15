# -*- coding: utf-8 -*-
"""電影字幕式螢幕疊加層:固定字幕框、只有文字浮在畫面上,其餘完全透明且滑鼠穿透。

排版規則(2026-08-16 正式版規格):
- 外框置中且位置固定,框內文字左對齊 → 增字不會左右漂移
- 永遠預留兩行高度、最多兩行:上行=已捲出的完成行(永不重排),下行=進行中
- 每行約 26 個中文字寬;超過時沿逗號/句號等語句邊界整行捲動
- 進行中尾端若是不完整的英文單字,先藏起來等整個單字完成
- 結束(stdin EOF)前最後一行至少停留 1.5 秒

由 live_subtitles.py 以子程序啟動,stdin 每行一個 JSON 事件:
  {"kind":"live","utt":3,"committed":"...","tail":"..."}  # 進行中:穩定前綴+浮動尾端(≤12字)
  {"kind":"final","utt":3,"text":"..."}                   # 句子定稿
單獨驗收:python overlay.py --demo / --test

視窗實作:Tk 色鍵透明 + WS_EX_TRANSPARENT(滑鼠穿透,連文字像素也不擋)
+ WS_EX_NOACTIVATE(永不搶焦點)+ WS_EX_TOOLWINDOW(不進工作列/Alt-Tab)。
"""
import argparse
import ctypes
import json
import queue
import re
import sys
import threading
import time
import tkinter as tk

KEY = "#0a0b0c"  # 色鍵:此色像素全透明(選極罕見暗色,避免與字幕色衝突)
HOLD = 5.0  # 無後續字幕時停留秒數
LINE_CHARS = 26  # 每行中文字寬(ASCII 算半字)
BREAKS = "，。、？！;；：,.?!"


def _w(s):
    """顯示寬度:CJK 1、ASCII 0.5。"""
    return sum(1 if ord(ch) > 0x2E80 else 0.5 for ch in s)


class Overlay:
    def __init__(self, font_size, bottom, width_ratio, read_stdin=True):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", KEY)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.line_h = font_size + 18
        self.h = self.line_h * 2 + 24  # 永遠預留兩行
        self.w = sw
        self.frame_w = min(int(font_size * (LINE_CHARS + 2)), int(sw * width_ratio))
        self.left = (sw - self.frame_w) // 2  # 外框置中、位置固定;框內左對齊
        self.root.geometry(f"{sw}x{self.h}+0+{sh - self.h - bottom}")
        self.canvas = tk.Canvas(self.root, bg=KEY, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.font = ("Microsoft JhengHei", font_size, "bold")
        # 顯示狀態:prev=已捲出的上一行(永不重排),cur=進行中行(穩定前綴),tail=浮動尾端
        self.prev = ""
        self.cur = ""
        self.tail = ""
        self.cur_utt = None
        self.consumed = 0  # 目前句的 committed 已吸收長度
        self.last_update = 0.0
        self.q = queue.Queue()
        self.dirty = True

        self.root.update_idletasks()
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
        style = user32.GetWindowLongW(hwnd, -20)
        # LAYERED | TRANSPARENT(滑鼠穿透) | NOACTIVATE | TOOLWINDOW
        user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20 | 0x8000000 | 0x80)
        self.hwnd = hwnd
        print(f"dbg hwnd={hwnd} screen={sw}x{sh} frame_w={self.frame_w} left={self.left}",
              file=sys.stderr, flush=True)
        if read_stdin:  # demo/test 模式不讀 stdin:非互動環境 stdin 立即 EOF 會誤觸自我銷毀
            threading.Thread(target=self._stdin_reader, daemon=True).start()
        self.root.after(50, self._tick)
        self.root.after(2000, self._assert_topmost)

    def _stdin_reader(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                self.q.put(json.loads(line))
            except json.JSONDecodeError:
                pass
        self.q.put({"kind": "eof"})

    def _assert_topmost(self):
        # 有些全螢幕/置頂應用會蓋上來,週期性重申最上層(不搶焦點)
        ctypes.windll.user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 0x10 | 0x2 | 0x1)
        self.root.after(2000, self._assert_topmost)

    # --- 排版核心 ---

    def _append(self, delta):
        """把新提交的文字接到進行中行;過長時沿語句邊界整行捲動(捲出的行不再重排)。"""
        self.cur += delta
        while _w(self.cur) + _w(self.tail) > LINE_CHARS and _w(self.cur) > 4:
            cut = None
            acc = 0.0
            for i, ch in enumerate(self.cur):
                acc += 1 if ord(ch) > 0x2E80 else 0.5
                if acc > LINE_CHARS:
                    break
                if ch in BREAKS:
                    cut = i + 1
            if cut is None or _w(self.cur[:cut]) < 6:  # 無語句邊界就硬切滿行
                cut = max(1, i)
            self.prev = self.cur[:cut].strip()
            self.cur = self.cur[cut:].lstrip()

    def _tick(self):
        while True:
            try:
                ev = self.q.get_nowait()
            except queue.Empty:
                break
            kind = ev.get("kind")
            if kind == "live":
                if ev.get("utt") != self.cur_utt:
                    self.cur_utt = ev.get("utt")
                    self.consumed = 0
                c = ev.get("committed", "")
                if len(c) > self.consumed:
                    self._append(c[self.consumed:])
                    self.consumed = len(c)
                self.tail = ev.get("tail", "")
                self.last_update = time.time()
            elif kind == "final":
                t = ev.get("text", "")
                if ev.get("utt") == self.cur_utt and len(t) > self.consumed:
                    self._append(t[self.consumed:])
                elif ev.get("utt") != self.cur_utt:
                    self._append(t)
                self.tail = ""
                self.cur_utt = None
                self.consumed = 0
                self.last_update = time.time()
            elif kind == "eof":
                # 收尾:最後一句至少停留 1.5s 再消失
                self.root.after(1500, self.root.destroy)
                return
            self.dirty = True
        if (self.prev or self.cur or self.tail) and time.time() - self.last_update > HOLD:
            self.prev = self.cur = self.tail = ""
            self.cur_utt = None
            self.consumed = 0
            self.dirty = True
        if self.dirty:
            self._redraw()
            self.dirty = False
        self.root.after(50, self._tick)

    def _text_line(self, text, y_bottom):
        """左對齊單行,白字黑邊(黑色 8 向偏移打底)。"""
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
            self.canvas.create_text(self.left + dx, y_bottom + dy, text=text, font=self.font,
                                    fill="black", anchor="sw")
        self.canvas.create_text(self.left, y_bottom, text=text, font=self.font,
                                fill="white", anchor="sw")

    def _redraw(self):
        self.canvas.delete("all")
        tail = self.tail
        # 尾端不顯示不完整的英文單字(mic wa → 等 wave 完整再上)
        m = re.search(r"[A-Za-z]+$", tail)
        if m:
            tail = tail[:m.start()].rstrip()
        cur_line = (self.cur + tail).strip()
        y2 = self.h - 12  # 下行(進行中)固定位置
        y1 = y2 - self.line_h  # 上行(完成行)固定位置
        if cur_line:
            self._text_line(cur_line, y2)
        if self.prev:
            self._text_line(self.prev, y1)

    def run(self):
        self.root.mainloop()


def demo_feed():
    lines = [
        {"kind": "live", "utt": 1, "committed": "那我們會", "tail": "以這個例"},
        {"kind": "live", "utt": 1, "committed": "那我們會以這個例子做情境,", "tail": "看 Orbit 怎"},
        {"kind": "live", "utt": 1, "committed": "那我們會以這個例子做情境,看 Orbit 怎麼陪漢神走這個檔期,", "tail": "首先是 Orbi"},
        {"kind": "final", "utt": 1, "text": "那我們會以這個例子做情境,看 Orbit 怎麼陪漢神走這個檔期,首先是 Orbit。"},
        {"kind": "live", "utt": 2, "committed": "到底能帶給漢神什麼效益,", "tail": "我們先用一頁"},
        {"kind": "final", "utt": 2, "text": "到底能帶給漢神什麼效益,我們先用一頁說明。"},
    ]
    time.sleep(1)
    for ev in lines:
        print(json.dumps(ev, ensure_ascii=False), flush=True)
        time.sleep(1.6)
    time.sleep(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font-size", type=int, default=28)
    ap.add_argument("--bottom", type=int, default=70, help="距螢幕底部像素")
    ap.add_argument("--width-ratio", type=float, default=0.78)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--test", action="store_true", help="立即顯示靜態測試字幕(渲染診斷用)")
    args = ap.parse_args()

    if args.test:
        ov = Overlay(args.font_size, args.bottom, args.width_ratio, read_stdin=False)
        ov.prev = "上一行:固定框內左對齊,永不重排。"
        ov.cur = "進行中行:白字黑邊 Orbit 一二三,"
        ov.tail = "浮動尾端"
        ov.dirty = True
        ov.run()
        return

    if args.demo:
        import subprocess
        feeder = subprocess.Popen(
            [sys.executable, "-c", "import overlay; overlay.demo_feed()"],
            stdout=subprocess.PIPE, cwd=str(__import__('pathlib').Path(__file__).parent),
            text=True, encoding="utf-8")
        ov = Overlay(args.font_size, args.bottom, args.width_ratio, read_stdin=False)

        def pump():
            for ln in feeder.stdout:
                ov.q.put(json.loads(ln))
            ov.q.put({"kind": "eof"})
        threading.Thread(target=pump, daemon=True).start()
        ov.run()
        return

    Overlay(args.font_size, args.bottom, args.width_ratio).run()


if __name__ == "__main__":
    main()
