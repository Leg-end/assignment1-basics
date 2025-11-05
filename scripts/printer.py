import shutil
import textwrap


class StreamPrinter:
    def __init__(self):
        self.terminal_width = shutil.get_terminal_size().columns
        self.current_line = ""
    
    def update(self, new_char):
        """纯流式打印，绝不重复"""
        if new_char == '\n':
            # 换行符：直接换行
            print('', flush=True)
            self.current_line = ""
            return
        
        self.current_line += new_char
        
        # 检查是否需要换行
        if len(self.current_line) >= self.terminal_width:
            # 直接打印整行并换行
            print(self.current_line, flush=True)
            self.current_line = ""
        else:
            # 实时显示当前行（只显示当前行）
            print(f"\r{self.current_line}", end="", flush=True)
    
    def complete(self):
        """完成输出"""
        # 打印最后一行（如果有）
        if self.current_line:
            print(self.current_line, flush=True)
        print()
        self.current_line = ""
