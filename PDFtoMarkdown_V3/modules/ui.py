class ConsoleProgress:

    def progress(self, value):
        pass

    def empty(self):
        pass


class ConsoleUI:

    def subheader(self, text):
        print(f"\n{text}")
        print("-" * len(text))

    def info(self, text):
        print(text)

    def success(self, text):
        print(f"[OK] {text}")

    def warning(self, text):
        print(f"[!] {text}")

    def write(self, text):
        print(text)

    def progress(self, value=0):
        return ConsoleProgress()