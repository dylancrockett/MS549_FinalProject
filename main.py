from calculator import Calculator
from tkinter import *

# calculator instance
calculator = Calculator()

# gui
root = Tk()
root.title("Calculator")

# colors
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

# font
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


# Send function
def calculate(*_):
    # get input expression
    expr = expression_input.get()

    # evaluate expression
    result = calculator.eval(expr)

    # append result to log
    txt.insert(END, "\n" + expr)
    txt.insert(END, "\n> " + str(result) + "\n")
    expression_input.delete(0, END)


# calculation log
txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
txt.grid(row=1, column=0, columnspan=2)

# scrollbar
scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)

# expression input
expression_input = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=52)
expression_input.grid(row=2, column=0)
expression_input.bind("<Return>", calculate)

# button to execute computation
compute = Button(root, text="Eval", font=FONT_BOLD, bg=BG_GRAY, command=calculate)
compute.grid(row=2, column=1)

# run loop
root.mainloop()
